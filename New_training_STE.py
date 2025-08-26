import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader


# ---------------------------
# 1) STE binarization blocks
# ---------------------------
class _BinarizeSTE(torch.autograd.Function):
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
    return _BinarizeSTE.apply(x, allow_scale, quant_mode)


class BinarizeConv2d(nn.Conv2d):
    """
    Conv2d that binarizes activations and weights during forward.
    Backward uses STE. BatchNorm stays float.
    """
    def __init__(self, *args, allow_scale=False, quant_mode='det', **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_scale = allow_scale
        self.quant_mode = quant_mode

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, allow_scale=False, quant_mode='det'):
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
        new.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data)
        return new

    def forward(self, x):
        x_b = binarize(x, allow_scale=False, quant_mode=self.quant_mode)
        w_b = binarize(self.weight, allow_scale=self.allow_scale, quant_mode=self.quant_mode)
        return F.conv2d(x_b, w_b, self.bias, self.stride, self.padding, self.dilation, self.groups)


# --------------------------------------
# 2) Convert MobileNetV2 into a BNN body
# --------------------------------------
def convert_mobilenetv2_to_bnn(mnet: nn.Module,
                               keep_first_conv_fp: bool = True,
                               quant_mode: str = 'det',
                               allow_scale: bool = False) -> nn.Module:
    """
    Replace every Conv2d in features with BinarizeConv2d.
    Classifier (final Linear) stays float.
    """
    first_conv_seen = [False]

    def _convert(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                if keep_first_conv_fp and not first_conv_seen[0]:
                    first_conv_seen[0] = True  # keep the very first image conv in float
                else:
                    setattr(module, name, BinarizeConv2d.from_conv(child,
                                                                  allow_scale=allow_scale,
                                                                  quant_mode=quant_mode))
            else:
                _convert(child)

    _convert(mnet.features)
    return mnet


# ---------------------------
# 3) Training
# ---------------------------
def train_model_quantized(
    data_is_quantized: bool = False,
    stage_epochs: tuple[int, int, int] = (2, 10, 12),
    early_stop_patience: int = 5,
    use_lr_on_plateau: bool = True,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
):
    """
    Progressive fine-tuning in 3 stages with a BNN feature extractor (classifier remains float).
    Includes per-epoch validation (ValLoss/ValAcc), optional ReduceLROnPlateau, and early stopping per stage.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained MobileNetV2, set 2-class head
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    # Convert features to BNN, keep classifier float
    mobilenet = convert_mobilenetv2_to_bnn(
        mobilenet,
        keep_first_conv_fp=True,   # set False if you want every conv binary
        quant_mode='det',
        allow_scale=False
    ).to(device)

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

        # Freeze all
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze requested layers
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
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for v_inputs, v_labels in val_loader:
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_labels)
                    val_loss += v_loss.item() * v_inputs.size(0)
                    val_correct += v_outputs.argmax(dim=1).eq(v_labels).sum().item()
                    val_total += v_labels.size(0)

            val_loss /= max(1, val_total)
            val_acc = val_correct / max(1, val_total)

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

            # ----- Early Stopping per stage by ValLoss -----
            if val_loss < best_val_loss_stage:
                best_val_loss_stage = val_loss
                early_stopping_counter = 0
                best_model_weights_stage = copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stop_patience:
                    print("Early stopping triggered for this stage.")
                    break

            # Save best (global) by ValAcc with your filenames rule
            if val_acc > best_val_acc_global:
                best_val_acc_global = val_acc
                best_path = "mobilenetv2_all_quantize.pth" if data_is_quantized else "mobilenetv2_model_quantize.pth"
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

    # Optional: also save the final (last) state if you want
    final_path = "mobilenetv2_all_quantize.pth" if data_is_quantized else "mobilenetv2_model_quantize.pth"
    torch.save(mobilenet.state_dict(), final_path)
    print(f"Final model saved as {final_path}")


if __name__ == "__main__":
    train_model_quantized()
