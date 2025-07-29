import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_preparation import prepare_data
import time

# Colors for terminal output
YELLOW = '\033[93m'
ENDC = '\033[0m'
RED = '\033[91m'

def finetune_feature18(num_epochs=5, lr=0.00005):
    """
    Fine-tunes only the features.18 block and classifier of MobileNetV2.
    Loads model from and saves back to 'mobilenetv2_binary.pth'.
    """

    print(f"{YELLOW}--- Starting Feature 18 Fine-tuning ---{ENDC}")

    # Load data
    train_loader, val_loader, test_loader, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{YELLOW}Using device: {device}{ENDC}")

    # Load the pretrained model
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    try:
        mobilenet.load_state_dict(torch.load("mobilenetv2_binary.pth", map_location=device))
        print(f"{YELLOW}Loaded weights from mobilenetv2_binary.pth{ENDC}")
    except Exception as e:
        print(f"{RED}Error loading pretrained model: {e}{ENDC}")
        return

    mobilenet = mobilenet.to(device)

    # Freeze all layers
    for param in mobilenet.parameters():
        param.requires_grad = False

    # Unfreeze only features.18 and classifier
    for name, param in mobilenet.named_parameters():
        if any(layer in name for layer in ["features.18", "classifier"]):
            param.requires_grad = True

    # Check how many parameters are being trained
    total_params = len(list(mobilenet.parameters()))
    trainable_params = sum(p.requires_grad for p in mobilenet.parameters())
    print(f"{YELLOW}Fine-tuning only features.18 and classifier ({trainable_params} out of {total_params} parameters).{ENDC}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet.parameters()), lr=lr)

    # Training loop
    print(f"{YELLOW}🚀 Starting fine-tuning for {num_epochs} epochs...{ENDC}")
    start_time = time.time()

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
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / total_samples
        acc = correct / total_samples
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"{YELLOW}✅ Fine-tuning complete. Total time: {total_time/60:.1f} minutes.{ENDC}")

    # Save updated model back to same file
    torch.save(mobilenet.state_dict(), "mobilenetv2_binary.pth")
    print(f"{YELLOW}📦 Updated model saved to mobilenetv2_binary.pth (overwritten){ENDC}")

if __name__ == "__main__":
    finetune_feature18()
