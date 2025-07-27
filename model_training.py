import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data

# צבעים להדפסה מודגשת
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'


def train_model(num_epochs=10):
    """
    Load data, define and train MobileNetV2 model with partial unfreezing for fine-tuning.
    """

    # Load DataLoaders
    train_loader, val_loader, test_loader, label_encoder = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{YELLOW}Using device: {device}{ENDC}")

    # Load pretrained MobileNetV2
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers first
    for param in mobilenet.parameters():
        param.requires_grad = False

    # Replace the classifier head
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    # Unfreeze specific deep layers for fine-tuning
    for name, param in mobilenet.named_parameters():
        if any(layer in name for layer in ["features.14", "features.15", "features.16", "features.17", "classifier"]):
            param.requires_grad = True

    # Move model to device
    mobilenet = mobilenet.to(device)

    # Check frozen status
    frozen = sum([not param.requires_grad for param in mobilenet.parameters()])
    total = len(list(mobilenet.parameters()))
    print(f"{YELLOW}✅ Frozen {frozen} out of {total} layers.{ENDC}")
    print(f"{YELLOW}Note: You should see LESS than 100% frozen now – because some layers were unfrozen for fine-tuning.{ENDC}")

    # Loss and optimizer (smaller learning rate for fine-tuning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet.parameters()), lr=0.0003)

    # Training loop
    print(f"\n{YELLOW}🚀 Starting training for {num_epochs} epochs...{ENDC}")
    start_time = time.time()

    losses = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        mobilenet.train()

        running_loss = 0.0
        correct = 0
        total = 0

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
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        est_total_time = epoch_time * num_epochs
        est_remaining = est_total_time - epoch_time * (epoch + 1)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Time: {epoch_time:.1f}s | Estimated time left: {est_remaining/60:.1f} min")

    total_time = time.time() - start_time
    print(f"\n{YELLOW}✅ Training complete. Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes).{ENDC}")
    print(f"{YELLOW}📦 Model saved to mobilenetv2_binary.pth{ENDC}")
    torch.save(mobilenet.state_dict(), "mobilenetv2_binary.pth")

    # Final loss trend summary
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print(f"\n{YELLOW}📉 Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f} — training is improving!{ENDC}")
    else:
        print(f"\n{RED}⚠️ Loss did NOT decrease — training might not be effective yet. Consider tuning learning rate or unfreezing more layers.{ENDC}")

    print(f"\n{RED}🧪 You can now run evaluation on test_loader if desired!{ENDC}")


if __name__ == "__main__":
    train_model(num_epochs=10)
