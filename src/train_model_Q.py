import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from model_validation import evaluate


# 1) Straight-Through Estimator (STE) binarization building blocks
class _BinarizeSTE(torch.autograd.Function):
    """
    Autograd function implementing binarization with a straight-through estimator.

    Forward: binarize to ±1, optionally scaled by max(|x|) to preserve magnitude (XNOR-style).
    Backward: pass gradients through unchanged (approximate derivative as 1 inside support).

    Args:
        allow_scale (bool): If True, multiply ±1 mask by max(|x|).
        quant_mode (str): 'det' for deterministic sign; otherwise injects uniform noise
            before rounding to mimic a stochastic path (regularization).
    """
    @staticmethod
    def forward(ctx, x, allow_scale=False, quant_mode='det'):
        scale = x.abs().max() if allow_scale else x.new_tensor(1.0)
        if quant_mode == 'det':
            return x.div(scale).sign().mul(scale)
        else:
            noise = torch.rand_like(x).add(-0.5)
            out = (
                x.div(scale).add_(1).div_(2).add_(noise)
                  .clamp_(0, 1).round_().mul_(2).add_(-1).mul_(scale)
            )
            return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator
        return grad_output, None, None


def binarize(x, allow_scale=False, quant_mode='det'):
    """Convenience wrapper for STE binarization used by layers below."""
    return _BinarizeSTE.apply(x, allow_scale, quant_mode)


class BinarizeConv2d(nn.Conv2d):
    """
    Conv2d variant that binarizes activations and weights during forward.

    - Activations: binarized without scaling.
    - Weights: binarized, optional max-abs scaling via allow_scale.
    - BatchNorm (if present elsewhere) remains float.

    Use `BinarizeConv2d.from_conv(conv, ...)` to clone shape/params from a pretrained Conv2d.
    """
    def __init__(self, *args, allow_scale=False, quant_mode='det', **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_scale = allow_scale
        self.quant_mode = quant_mode

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, allow_scale=False, quant_mode='det'):
        """Clone hyperparameters and parameters from an existing Conv2d."""
        new = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
            allow_scale=allow_scale,
            quant_mode=quant_mode,
        )
        # Copy parameters to preserve initialization / pretrained state
        new.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data)
        return new

    def forward(self, x):
        # Binarize activations (no scaling) and weights (optionally scaled)
        x_b = binarize(x, allow_scale=False, quant_mode=self.quant_mode)
        w_b = binarize(self.weight, allow_scale=self.allow_scale, quant_mode=self.quant_mode)
        # Apply convolution with binarized weights
        return F.conv2d(x_b, w_b, self.bias, self.stride, self.padding, self.dilation, self.groups)


# 2) Convert MobileNetV2 feature extractor into a binarized (BNN) body
def convert_mobilenetv2_to_bnn(mnet: nn.Module,
                               keep_first_conv_fp: bool = True,
                               quant_mode: str = 'det',
                               allow_scale: bool = False) -> nn.Module:
    """
    Replace Conv2d layers inside `mnet.features` with `BinarizeConv2d`.

    - If keep_first_conv_fp=True, the very first stem conv remains float to preserve
      low-level features and stabilize training.
    - The classifier (final Linear) is not modified here.
    """
    first_conv_seen = [False] # mutable flag closed over by _convert

    def _convert(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                if keep_first_conv_fp and not first_conv_seen[0]:
                    # Keep the very first Conv2d (input stem) in float
                    first_conv_seen[0] = True  # keep the very first image conv in float
                else:
                    # Replace with binarized version (clone shape/params)
                    setattr(module, name, BinarizeConv2d.from_conv(child,
                                                                  allow_scale=allow_scale,
                                                                  quant_mode=quant_mode))
            else:
                _convert(child)

    _convert(mnet.features)
    return mnet


# 3) Training loop with progressive unfreezing and early stopping
def train_model_quantized(
    data_is_quantized: bool = False,
    stage_epochs: tuple[int, int, int] = (2, 10, 12),
    early_stop_patience: int = 5,
    use_lr_on_plateau: bool = True,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
):
    """
    Progressive fine-tuning in 3 stages using a binarized feature extractor; classifier stays float.

    - Per-epoch validation with ValLoss/ValAcc.
    - Optional ReduceLROnPlateau on validation loss.
    - Early stopping per stage based on validation loss.

    Args:
        data_is_quantized (bool): Used to pick best-checkpoint filename for bookkeeping.
        stage_epochs (tuple[int,int,int]): Epochs per stage: (classifier-only, last blocks, deeper blocks).
        early_stop_patience (int): Stop current stage after N non-improving epochs (ValLoss).
        use_lr_on_plateau (bool): Enable ReduceLROnPlateau on validation loss.
        train_loader (DataLoader): Yields (images, labels). Assumes preprocessing done upstream.
        val_loader (DataLoader): Yields (images, labels). Assumes preprocessing done upstream.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained MobileNetV2, set 2-class head
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    # Convert features to BNN, keep classifier float
    mobilenet = convert_mobilenetv2_to_bnn(
        mobilenet,
        keep_first_conv_fp=True,   # set False to binarize every conv layer
        quant_mode='det',
        allow_scale=False
    ).to(device)
    # Stage plan: progressively unfreeze more of the feature extractor
    stages = [
        {"layers": ["classifier"], "epochs": stage_epochs[0], "lr": 1e-3,  "name": "Stage 1: Classifier only"},
        {"layers": ["features.15", "features.16", "features.17", "features.18", "classifier"],
         "epochs": stage_epochs[1], "lr": 1e-4,  "name": "Stage 2: Features.15-18 + Classifier"},
        {"layers": ["features.8","features.9","features.10","features.11","features.12","features.13","features.14",
                    "features.15","features.16","features.17","features.18","classifier"],
         "epochs": stage_epochs[2], "lr": 5e-5, "name": "Stage 3: Features.8–18 + Classifier"},
    ]

    total_epochs = sum(s["epochs"] for s in stages)
    completed_epochs = 0
    global_start = time.time()

    # Track best globally by ValAcc
    best_val_acc_global = 0.0

    def run_training(model, stage, stage_idx):
        nonlocal completed_epochs, best_val_acc_global

        best_model_weights_stage = None
        best_val_loss_stage = float("inf")
        early_stopping_counter = 0

        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze parameters whose name contains any of the stage layer tokens
        for name, p in model.named_parameters():
            if any(layer in name for layer in stage["layers"]):
                p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=stage["lr"])
        scheduler = None
        if use_lr_on_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2, cooldown=1, min_lr=1e-6, verbose=False
            )

        print(f"\n--- {stage['name']} ---")
        print(f"LR={stage['lr']}, Epochs={stage['epochs']}")

        stage_start = time.time()

        for epoch in range(stage["epochs"]):
            epoch_start = time.time()
            model.train()

            running_loss, correct, total_samples = 0.0, 0, 0

            # -------- Train --------
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                correct += out.argmax(1).eq(yb).sum().item()
                total_samples += yb.size(0)

            train_loss = running_loss / max(1, total_samples)
            train_acc = correct / max(1, total_samples)

            # ----- Validation -----
            val_acc, val_loss = evaluate(model, val_loader, device, name=f"Stage {stage_idx+1} Epoch {epoch+1}")
            # Step LR scheduler based on val loss
            if scheduler is not None:
                scheduler.step(val_loss)

            # Progress / ETA
            completed_epochs += 1
            elapsed = time.time() - global_start
            avg_epoch_time = elapsed / max(1, completed_epochs)
            est_remaining = avg_epoch_time * (total_epochs - completed_epochs)

            print(
                f"Stage {stage_idx+1}/{len(stages)} | Epoch {epoch+1}/{stage['epochs']} "
                f"(Global {completed_epochs}/{total_epochs}) "
                f"| TrainLoss: {train_loss:.4f} | TrainAcc: {train_acc:.4f} "
                f"| ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} "
                f"| Time: {time.time() - epoch_start:.1f}s | ETA: {est_remaining/60:.1f} min"
            )

            # Early Stopping per stage by ValLoss
            if val_loss < best_val_loss_stage:
                best_val_loss_stage = val_loss
                early_stopping_counter = 0
                best_model_weights_stage = copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stop_patience:
                    print("Early stopping triggered for this stage.")
                    break

            # Save best-by-accuracy across all stages
            if val_acc > best_val_acc_global:
                best_val_acc_global = val_acc
                best_path = "mobilenetv2_best_fully_quantized.pth" if data_is_quantized else "mobilenetv2_best_model_quantized.pth"
                torch.save(model.state_dict(), best_path)
                print(f"New best validation accuracy. Saved: {best_path}")

        # Restore best weights for this stage (if any)
        if best_model_weights_stage is not None:
            model.load_state_dict(best_model_weights_stage)

        print(f"{stage['name']} finished in {(time.time() - stage_start)/60:.1f} min")

    # -------- Run all stages --------
    for i, stage in enumerate(stages):
        run_training(mobilenet, stage, i)

    total_time = time.time() - global_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes total.")

    # Run final validation on the trained model
    print("\n" + "="*50)
    print("FINAL VALIDATION")
    print("="*50)
    evaluate(mobilenet, val_loader, device, name="Final Validation")


if __name__ == "__main__":
    # Entry point: uses default args and expects train/val loaders provided by caller
    train_model_quantized()
