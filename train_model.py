import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data

# צבעים להדגשה
RED = '\033[91m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

def train_model(num_epochs=10):
    """
    Train MobileNetV2 with more unfrozen layers (mid + deep) for better fine-tuning.
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

    # Unfreeze **more layers** (mid + deep + classifier)
    for name, param in mobilenet.named_parameters():
        if any(layer in name for layer in [
            "features.10", "features.11", "features.12", "features.13",
            "features.14", "features.15", "features.16", "features.17",
            "classifier"
        ]):
            param.requires_grad = True

    # Move to device
    mobilenet = mobilenet.to(device)

    # Check frozen status
    frozen = sum([not param.requires_grad for param in mobilenet.parameters()])
    total = len(list(mobilenet.parameters()))
    print(f"{YELLOW}✅ Frozen {frozen} out of {total} layers.{ENDC}")
    print(f"{YELLOW}Note: Even fewer layers are frozen now — more layers are fine-tuning!{ENDC}")

    # Loss and optimizer (lower LR for sensitive fine-tuning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet.parameters()), lr=0.0002)

    # Training loop
    print(f"\n{YELLOW}🚀 Starting training for {num_epochs} epochs with deeper fine-tuning...{ENDC}")
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
        print(f"\n{YELLOW}📉 Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f} — deeper fine-tuning improved training!{ENDC}")
    else:
        print(f"\n{RED}⚠️ Loss did NOT decrease — consider even more epochs or different LR.{ENDC}")

    print(f"\n{RED}🧪 You can now run evaluation on test_loader if desired!{ENDC}")


def continue_training(num_epochs=10, lr=0.0002):
    # Load DataLoaders
    train_loader, val_loader, test_loader, label_encoder = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{YELLOW}Using device: {device}{ENDC}")

    # Load the model you already trained
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)
    mobilenet.load_state_dict(torch.load("mobilenetv2_binary.pth", map_location=device))
    mobilenet = mobilenet.to(device)

    # Freeze all layers first
    for param in mobilenet.parameters():
        param.requires_grad = False

    # Keep the same layers unfrozen (features.10–17 + classifier)
    for name, param in mobilenet.named_parameters():
        if any(layer in name for layer in [
            "features.10", "features.11", "features.12", "features.13",
            "features.14", "features.15", "features.16", "features.17",
            "classifier"
        ]):
            param.requires_grad = True

    # Check frozen status
    frozen = sum([not p.requires_grad for p in mobilenet.parameters()])
    total = len(list(mobilenet.parameters()))
    print(f"{YELLOW}✅ Continuing training with {total - frozen} layers trainable.{ENDC}")

    # Loss and optimizer (lower LR for fine-tuning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet.parameters()), lr=lr)

    # Training loop
    print(f"\n{YELLOW}🚀 Continuing fine-tuning for {num_epochs} more epochs...{ENDC}")
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
    print(f"\n{YELLOW}✅ Fine-tuning complete. Total time: {total_time/60:.1f} minutes.{ENDC}")

    # Save updated model
    torch.save(mobilenet.state_dict(), "mobilenetv2_binary.pth")
    print(f"{YELLOW}📦 Updated model saved to mobilenetv2_binary.pth{ENDC}")

    # Loss trend
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print(f"\n{YELLOW}📉 Loss improved further from {losses[0]:.4f} → {losses[-1]:.4f}!{ENDC}")
    else:
        print(f"\n{YELLOW}ℹ️ Loss trend stable, might need more layers or more epochs next.{ENDC}")



def deep_finetune(num_epochs=10, lr=0.0001):
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

    # Unfreeze MORE layers now: features.8–17 + classifier
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
    print(f"{YELLOW}✅ Deep fine-tuning with {total - frozen} layers trainable now.{ENDC}")

    # Loss and optimizer (low LR for safe fine-tuning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet.parameters()), lr=lr)

    # Training loop
    print(f"\n{YELLOW}🚀 Deep fine-tuning for {num_epochs} more epochs...{ENDC}")
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
    print(f"\n{YELLOW}✅ Deep fine-tuning complete. Total time: {total_time/60:.1f} minutes.{ENDC}")

    # Save updated model
    torch.save(mobilenet.state_dict(), "mobilenetv2_binary.pth")
    print(f"{YELLOW}📦 Updated model saved to mobilenetv2_binary.pth{ENDC}")

    # Loss trend summary
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print(f"\n{YELLOW}📉 Loss improved further from {losses[0]:.4f} → {losses[-1]:.4f}!{ENDC}")
    else:
        print(f"\n{YELLOW}ℹ️ Loss trend stable, next step could be even earlier layers or class weights.{ENDC}")

    print(f"\n{YELLOW}👉 After this finishes, run model_test.py again to check if Inflammatory improved!{ENDC}")


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

def quantize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. Load trained model ===
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    mobilenet.load_state_dict(torch.load("mobilenetv2_binary.pth", map_location=device))
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    print("✅ Original model loaded")

    # === 2. Apply dynamic quantization ===
    quantized_model = torch.quantization.quantize_dynamic(
        mobilenet, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "mobilenetv2_binary_quantized.pth")
    print("✅ Quantized model saved")


if __name__ == "__main__":
    train_model(num_epochs=10)
    continue_training(num_epochs=10, lr=0.0002)
    deep_finetune(num_epochs=10, lr=0.0001)

    # You can tweak weight_factor (1.2–1.5) if needed
    weighted_finetune(num_epochs=8, lr=0.0001, weight_factor=1.3)
    finetune_feature18()
    quantize()

