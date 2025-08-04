import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data

YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"


def train_model():
    """
    Progressive fine-tuning of MobileNetV2 in 3 stages:
    1. Train classifier only
    2. Unfreeze deeper blocks (features.8–17 + classifier)
    3. Unfreeze final block (features.18 + classifier) for fine polishing
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
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)
    mobilenet = mobilenet.to(device)

    # -------------------
    # Stage configs
    # -------------------
    stages = [
        {"layers": ["classifier"], "epochs": 10, "lr": 0.001, "name": "Stage 1: Classifier only"},
        {"layers": [
            "features.8", "features.9", "features.10", "features.11",
            "features.12", "features.13", "features.14", "features.15",
            "features.16", "features.17", "classifier"
        ], "epochs": 12, "lr": 0.0001, "name": "Stage 2: Features.8–17 + Classifier"},
        {"layers": ["features.18", "classifier"], "epochs": 6, "lr": 0.00005, "name": "Stage 3: Features.18 + Classifier"},
    ]

    total_epochs = sum(stage["epochs"] for stage in stages)
    completed_epochs = 0
    global_start = time.time()

    # -------------------
    # Training loop helper
    # -------------------
    def run_training(model, stage, stage_idx):
        nonlocal completed_epochs

        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze requested layers
        for name, param in model.named_parameters():
            if any(layer in name for layer in stage["layers"]):
                param.requires_grad = True

        # Optimizer + loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=stage["lr"])

        # Count trainable
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

            # Update counters
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

    # -------------------
    # Run all stages
    # -------------------
    for i, stage in enumerate(stages):
        run_training(mobilenet, stage, i)

    total_time = time.time() - global_start
    print(f"\n{YELLOW}🏁 Training complete in {total_time/60:.1f} minutes total.{ENDC}")

    # -------------------
    # Save final model
    # -------------------
    torch.save(mobilenet.state_dict(), "mobilenetv2_binary.pth")
    print(f"{YELLOW}📦 Final model saved as mobilenetv2_binary.pth{ENDC}")

if __name__ == "__main__":
    train_model()