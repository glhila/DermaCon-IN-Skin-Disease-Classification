import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data

YELLOW = "\033[93m"
ENDC = "\033[0m"

def weighted_finetune(num_epochs=8, lr=0.0001, weight_factor=1.3):
    # Load DataLoaders
    train_loader, val_loader, test_loader, label_encoder = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{YELLOW}Using device: {device}{ENDC}")

    # Load the current trained model
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)
    mobilenet.load_state_dict(torch.load("mobilenetv2_binary.pth", map_location=device))
    mobilenet = mobilenet.to(device)

    # Freeze all layers first
    for param in mobilenet.parameters():
        param.requires_grad = False

    # Unfreeze the same deeper layers (features.8–17 + classifier)
    for name, param in mobilenet.named_parameters():
        if any(layer in name for layer in [
            "features.8", "features.9",
            "features.10", "features.11", "features.12",
            "features.13", "features.14", "features.15",
            "features.16", "features.17",
            "classifier"
        ]):
            param.requires_grad = True

    # Check frozen vs trainable layers
    frozen = sum([not p.requires_grad for p in mobilenet.parameters()])
    total = len(list(mobilenet.parameters()))
    print(f"{YELLOW}✅ Weighted fine-tuning with {total - frozen} layers trainable now.{ENDC}")

    # Class weights → give higher weight to "Inflammatory"
    class_weights = torch.tensor([1.0, weight_factor]).to(device)
    print(f"{YELLOW}Using class weights: Infectious=1.0, Inflammatory={weight_factor}{ENDC}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet.parameters()), lr=lr)

    # Training loop
    print(f"\n{YELLOW}🚀 Weighted fine-tuning for {num_epochs} epochs...{ENDC}")
    start_time = time.time()
    losses = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        mobilenet.train()

        running_loss = 0.0
        correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = mobilenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct / total_samples
        losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        est_remaining = epoch_time * (num_epochs - epoch - 1)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Time: {epoch_time:.1f}s | Estimated time left: {est_remaining/60:.1f} min")

    total_time = time.time() - start_time
    print(f"\n{YELLOW}✅ Weighted fine-tuning complete. Total time: {total_time/60:.1f} minutes.{ENDC}")

    # Save updated model
    torch.save(mobilenet.state_dict(), "mobilenetv2_binary.pth")
    print(f"{YELLOW}📦 Updated model saved to mobilenetv2_binary.pth{ENDC}")

    # Loss trend summary
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print(f"\n{YELLOW}📉 Loss improved further from {losses[0]:.4f} → {losses[-1]:.4f}!{ENDC}")
    else:
        print(f"\n{YELLOW}ℹ️ Loss trend stable, next step could be even more class balance tuning or augmentations.{ENDC}")

    print(f"\n{YELLOW}👉 After this finishes, run model_test.py again to check if Inflammatory recall improved!{ENDC}")

if __name__ == "__main__":
    # You can tweak weight_factor (1.2–1.5) if needed
    weighted_finetune(num_epochs=8, lr=0.0001, weight_factor=1.3)
