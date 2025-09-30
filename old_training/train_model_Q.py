import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data

# QAT imports
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"

def train_model():
    """
    Quantization-Aware Training (QAT) + Progressive fine-tuning of MobileNetV2 in 3 stages:
      1) Train classifier only
      2) Unfreeze deeper blocks (features.15–18 + classifier)
      3) Unfreeze features.8–18 + classifier
    During training, fake-quant + observers simulate INT8. After training, we convert
    to a real INT8 model and save BOTH the QAT checkpoint and the converted model.
    """

    # -------------------
    # Data + device setup
    # -------------------
    train_loader, val_loader, test_loader, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{YELLOW}Using device: {device}{ENDC}")

    # -------------------
    # Load pretrained base
    # -------------------
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    # -------------------
    # QAT config (FX)
    # -------------------
    # Use "fbgemm" for x86 (Windows/Linux). If targeting ARM/mobile, set to "qnnpack".
    torch.backends.quantized.engine = "fbgemm"
    qat_qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
    qconfig_mapping = QConfigMapping().set_global(qat_qconfig)

    # Prepare for QAT (insert observers + fake-quant)
    model = model.to(device).train()
    example_input = torch.randn(1, 3, 224, 224).to(device)  # match your input size
    model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)

    # -------------------
    # Stage configs
    # -------------------
    stages = [
        {"layers": ["classifier"], "epochs": 2, "lr": 1e-3,   "name": "Stage 1: Classifier only"},
        {"layers": ["features.15", "features.16", "features.17", "features.18", "classifier"],
         "epochs": 10, "lr": 1e-4,  "name": "Stage 2: Features.15-18 + Classifier"},
        {"layers": [
            "features.8", "features.9", "features.10", "features.11",
            "features.12", "features.13", "features.14", "features.15",
            "features.16", "features.17", "features.18", "classifier"
        ], "epochs": 12, "lr": 5e-5, "name": "Stage 3: Features.8–18 + Classifier"}
    ]

    total_epochs = sum(s["epochs"] for s in stages)
    completed_epochs = 0
    global_start = time.time()

    # -------------- helpers --------------
    def is_target_param(name: str, wanted_layers):
        # FX may change '.' to '_' in GraphModule names; allow both.
        name_us = name.replace('.', '_')
        for layer in wanted_layers:
            if layer in name or layer.replace('.', '_') in name_us:
                return True
        return False

    def set_trainable_params(m, wanted_layers):
        for p in m.parameters():
            p.requires_grad = False
        for n, p in m.named_parameters():
            if is_target_param(n, wanted_layers):
                p.requires_grad = True

    def count_trainable(m):
        total, trainable = 0, 0
        for p in m.parameters():
            total += 1
            trainable += int(p.requires_grad)
        return trainable, total

    def disable_observers(m):
        # Stop updating activation/weight ranges (freeze quant scales)
        for mod in m.modules():
            if hasattr(mod, "activation_post_process") and mod.activation_post_process is not None:
                try:
                    mod.activation_post_process.disable_observer()
                except Exception:
                    pass

    def freeze_bn_stats(m):
        # Freeze BatchNorm running stats near the end for stability
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mod.eval()

    # -------------------
    # Training loop helper
    # -------------------
    def run_training(model, stage, stage_idx):
        nonlocal completed_epochs

        # Freeze/unfreeze for this stage
        set_trainable_params(model, stage["layers"])

        # Optimizer + loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=stage["lr"])

        # Count trainable
        trn, tot = count_trainable(model)
        print(f"\n{YELLOW}--- {stage['name']} ---")
        print(f"Trainable params: {trn}/{tot}, LR={stage['lr']}, Epochs={stage['epochs']}{ENDC}")

        stage_start = time.time()
        for epoch in range(stage["epochs"]):
            epoch_start = time.time()
            model.train()

            running_loss, correct, total_samples = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += outputs.argmax(dim=1).eq(labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = running_loss / max(1, total_samples)
            acc = correct / max(1, total_samples)
            epoch_time = time.time() - epoch_start

            # QAT schedules
            completed_epochs += 1
            if completed_epochs == int(0.7 * total_epochs):
                print(f"{YELLOW}🔒 Disabling observers (freeze quant ranges){ENDC}")
                disable_observers(model)
            if completed_epochs == int(0.9 * total_epochs):
                print(f"{YELLOW}❄️  Freezing BatchNorm running stats{ENDC}")
                freeze_bn_stats(model)

            # ETA
            elapsed = time.time() - global_start
            avg_epoch_time = elapsed / max(1, completed_epochs)
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

    # -------------------
    # Run all stages (QAT)
    # -------------------
    for i, stage in enumerate(stages):
        run_training(model, stage, i)

    total_time = time.time() - global_start
    print(f"\n{YELLOW}🏁 Training complete in {total_time/60:.1f} minutes total.{ENDC}")

    # -------------------
    # Save QAT checkpoint (trainable: float weights + fake-quant + observers)
    # -------------------
    torch.save(model.state_dict(), "extra/mobilenetv2_qat_state.pth")
    print(f"{YELLOW}📦 QAT checkpoint saved as mobilenetv2_qat_state.pth{ENDC}")

    # -------------------
    # Convert to INT8 for inference + save FULL MODEL (recommended)
    # -------------------
    model.eval()
    int8_model = convert_fx(model)
    torch.save(int8_model, "extra/mobilenetv2_int8.pt")  # full module, easy to load
    print(f"{YELLOW}🧊 INT8 model saved as mobilenetv2_int8.pt{ENDC}")

    # (Optional) also save the INT8 state_dict if you want:
    torch.save(int8_model.state_dict(), "mobilenetv2_model_quantized.pth")
    print(f"{YELLOW}🧊 INT8 state_dict saved as mobilenetv2_model_quantized.pth{ENDC}")

if __name__ == "__main__":
    train_model()
