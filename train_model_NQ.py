import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader



def train_model(
    data_is_quantized: bool = False,
    stage_epochs: tuple[int, int, int] = (2, 10, 12),
    early_stop_patience: int = 5,
    use_lr_on_plateau: bool = True,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None
):
    """
    Progressive fine-tuning of MobileNetV2 in 3 stages:
    1) Train classifier only
    2) Unfreeze deeper blocks (features.15–18 + classifier)
    3) Unfreeze final block (features.8–18 + classifier) for fine polishing
    Includes per-epoch validation and early stopping.
    """

    # -------------------
    # Device
    # -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------
    # Model
    # -------------------
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    # -------------------
    # Stages
    # -------------------
    stages = [
        {"layers": ["classifier"], "epochs": stage_epochs[0], "lr": 0.001, "name": "Stage 1: Classifier only"},
        {"layers": ["features", "classifier"], "epochs": stage_epochs[1], "lr": 0.0001,
         "name": "Stage 2: Features + Classifier"},
        {"layers": ["features", "classifier"], "epochs": stage_epochs[2], "lr": 0.00005,
         "name": "Stage 3: Features + Classifier"}
    ]

    total_epochs = sum(s["epochs"] for s in stages)
    completed_epochs = 0
    global_start = time.time()

    # Track best across all stages
    best_val_acc_global = 0.0

    # -------------------
    # Training loop helper
    # -------------------
    def run_training(model, stage, stage_idx):
        nonlocal completed_epochs, best_val_acc_global

        best_model_weights_stage = None
        best_val_loss_stage = float("inf")
        early_stopping_counter = 0
        patience = early_stop_patience

        # Freeze all
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze requested
        for name, p in model.named_parameters():
            if any(layer in name for layer in stage["layers"]):
                p.requires_grad = True

        # Optimizer / loss / scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=stage["lr"])
        scheduler = None
        if use_lr_on_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2, cooldown=1, min_lr=1e-6,
            )

        # Info
        frozen = sum(int(not p.requires_grad) for p in model.parameters())
        total = sum(1 for _ in model.parameters())
        print(f"\n--- {stage['name']} ---")
        print(f"Trainable params (tensors): {total - frozen}/{total}, LR={stage['lr']}, Epochs={stage['epochs']}")

        stage_start = time.time()

        for epoch in range(stage["epochs"]):
            epoch_start = time.time()
            model.train()

            running_loss, correct, total_samples = 0.0, 0, 0

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
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    val_correct += out.argmax(1).eq(yb).sum().item()
                    val_total += yb.size(0)

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

            # ----- Early stopping (per stage by val_loss) -----
            if val_loss < best_val_loss_stage:
                best_val_loss_stage = val_loss
                early_stopping_counter = 0
                best_model_weights_stage = copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered for this stage.")
                    break

            # Save best-by-accuracy across all stages
            if val_acc > best_val_acc_global:
                best_val_acc_global = val_acc
                torch.save(
                    model.state_dict(),
                    "mobilenetv2_best_data_quantized.pth" if data_is_quantized else "mobilenetv2_best_not_quantized.pth",
                )
                print(
                    "New best validation accuracy. "
                    f"Saved: {'mobilenetv2_best_data_quantized.pth' if data_is_quantized else 'mobilenetv2_best_not_quantized.pth'}"
                )

        # Restore best weights for this stage (if any)
        if best_model_weights_stage is not None:
            model.load_state_dict(best_model_weights_stage)

        print(f"{stage['name']} finished in {(time.time() - stage_start)/60:.1f} min")

    # -------------------
    # Run all stages
    # -------------------
    for i, stage in enumerate(stages):
        run_training(model, stage, i)

    total_time = time.time() - global_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes total.")

    # Optionally return the trained model
    return model


if __name__ == "__main__":
    train_model()
