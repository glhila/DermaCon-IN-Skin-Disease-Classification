import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import time
from data_preparation import prepare_data

YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"

# ---------------------------
# 1) STE binarization blocks
# ---------------------------
class _BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, allow_scale=False, quant_mode='det'):
        # Optional per-tensor scale (kept simple: max-abs). Often set to 1 in practice.
        scale = x.abs().max() if allow_scale else x.new_tensor(1.0)

        if quant_mode == 'det':
            # deterministic sign
            return x.div(scale).sign().mul(scale)
        else:
            # stochastic (rarely used)
            noise = torch.rand_like(x).add(-0.5)  # [-0.5, 0.5]
            out = x.div(scale).add_(1).div_(2).add_(noise).clamp_(0, 1).round_().mul_(2).add_(-1).mul_(scale)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients as-is
        return grad_output, None, None

def binarize(x, allow_scale=False, quant_mode='det'):
    return _BinarizeSTE.apply(x, allow_scale, quant_mode)


class BinarizeConv2d(nn.Conv2d):
    """
    Drop-in Conv2d that:
      - Binarizes input activations to {-1,+1} (forward only)
      - Binarizes weights to {-1,+1} (forward only)
      - Uses STE in backward (implemented in _BinarizeSTE)
    BatchNorm stays in float outside this layer as usual.
    """
    def __init__(self, *args, allow_scale=False, quant_mode='det', **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_scale = allow_scale
        self.quant_mode = quant_mode

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, allow_scale=False, quant_mode='det'):
        # Create a binarized version copying hyperparams/weights
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
        # Copy weights/bias
        new.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data)
        return new

    def forward(self, x):
        # 1) binarize activations
        x_b = binarize(x, allow_scale=False, quant_mode=self.quant_mode)
        # 2) binarize weights (per-layer)
        w_b = binarize(self.weight, allow_scale=self.allow_scale, quant_mode=self.quant_mode)
        # 3) conv in float math but with +-1 tensors
        return F.conv2d(
            x_b, w_b, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

# --------------------------------------
# 2) Convert MobileNetV2 into a BNN body
# --------------------------------------
def convert_mobilenetv2_to_bnn(mnet: nn.Module,
                               keep_first_conv_fp: bool = True,
                               quant_mode: str = 'det',
                               allow_scale: bool = False) -> nn.Module:
    """
    Replaces every Conv2d in 'features' with BinarizeConv2d.
    Keeps classifier (final Linear) real-valued.
    Optionally keeps the very first image conv in float for stability.
    """
    first_conv_seen = [False]

    def _convert(module: nn.Module):
        for name, child in list(module.named_children()):
            # Replace Conv2d everywhere (features), except the very first one if requested
            if isinstance(child, nn.Conv2d):
                if keep_first_conv_fp and not first_conv_seen[0]:
                    first_conv_seen[0] = True
                    # keep original float conv
                else:
                    setattr(module, name, BinarizeConv2d.from_conv(child,
                                                                  allow_scale=allow_scale,
                                                                  quant_mode=quant_mode))
            else:
                _convert(child)

    # Only binarize the feature extractor; leave classifier real
    _convert(mnet.features)
    return mnet

# ---------------------------
# 3) Training (unchanged API)
# ---------------------------
def train_model():
    """
    Progressive fine-tuning in 3 stages, exactly like your original,
    but with a BNN feature extractor (classifier remains float).
    """

    # Data + device
    train_loader, val_loader, test_loader, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{YELLOW}Using device: {device}{ENDC}")

    # Load pretrained MobileNetV2, set 2-class head
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    # >>> Convert to BNN: all feature Convs are binarized, classifier stays real <<<
    mobilenet = convert_mobilenetv2_to_bnn(
        mobilenet,
        keep_first_conv_fp=True,  # set False if you want *every* conv binary
        quant_mode='det',
        allow_scale=False
    )

    mobilenet = mobilenet.to(device)

    # Same 3-stage plan you had
    stages = [
        {"layers": ["classifier"], "epochs": 2, "lr": 0.001, "name": "Stage 1: Classifier only"},
        {"layers": ["features.15", "features.16", "features.17", "features.18", "classifier"],
         "epochs": 10, "lr": 0.0001, "name": "Stage 2: Features.15-18 + Classifier "},
        {"layers": [
            "features.8", "features.9", "features.10", "features.11",
            "features.12", "features.13", "features.14", "features.15",
            "features.16", "features.17", "features.18", "classifier"
        ], "epochs": 12, "lr": 0.00005, "name": "Stage 3: Features.8–18 + Classifier"}
    ]

    total_epochs = sum(stage["epochs"] for stage in stages)
    completed_epochs = 0
    global_start = time.time()

    def run_training(model, stage, stage_idx):
        nonlocal completed_epochs

        # Freeze all
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze requested layers
        for name, p in model.named_parameters():
            if any(layer in name for layer in stage["layers"]):
                p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=stage["lr"])

        frozen = sum([not p.requires_grad for p in model.parameters()])
        total = len(list(model.parameters()))
        print(f"\n{YELLOW}--- {stage['name']} ---")
        print(f"Trainable params: {total - frozen}/{total}, LR={stage['lr']}, Epochs={stage['epochs']}{ENDC}")

        stage_start = time.time()
        for epoch in range(stage["epochs"]):
            epoch_start = time.time()
            model.train()

            running_loss, correct, total_samples = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += outputs.argmax(dim=1).eq(labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = running_loss / total_samples
            acc = correct / total_samples
            epoch_time = time.time() - epoch_start

            completed_epochs += 1
            elapsed = time.time() - global_start
            avg_epoch_time = elapsed / completed_epochs
            est_remaining = avg_epoch_time * (total_epochs - completed_epochs)

            print(
                f"Stage {stage_idx+1}/{len(stages)} | Epoch {epoch+1}/{stage['epochs']} "
                f"(Global {completed_epochs}/{total_epochs}) "
                f"| Loss: {avg_loss:.4f} | Acc: {acc:.4f} "
                f"| Time: {epoch_time:.1f}s "
                f"| ETA left: {est_remaining/60:.1f} min"
            )

        stage_time = time.time() - stage_start
        print(f"{YELLOW}✅ {stage['name']} finished in {stage_time/60:.1f} min{ENDC}")

    # Run stages
    for i, stage in enumerate(stages):
        run_training(mobilenet, stage, i)

    total_time = time.time() - global_start
    print(f"\n{YELLOW}🏁 Training complete in {total_time/60:.1f} minutes total.{ENDC}")

    # Save
    torch.save(mobilenet.state_dict(), "STE_model.pth")
    print(f"{YELLOW}📦 Final model saved as STE_model.pth{ENDC}")


if __name__ == "__main__":
    train_model()
