import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data

# צבעים להדפסות
YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"


# --- טעינת מודל בסיסי ---
def load_model(device, quantized=False):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    # טוענים משקלים אם קיימים
    try:
        if quantized:
            model.load_state_dict(torch.load("../mobilenetv2_NOT_quantized.pth", map_location=device))
            print(f"{YELLOW}📦 Loaded quantized weights from mobilenetv2_NOT_quantized.pth{ENDC}")
        else:
            model.load_state_dict(torch.load("mobilenetv2_binary.pth", map_location=device))
            print(f"{YELLOW}📦 Loaded weights from mobilenetv2_binary.pth{ENDC}")
    except FileNotFoundError:
        print(f"{RED}⚠️ No existing weights found, starting fresh!{ENDC}")

    return model.to(device)


# --- אימון כללי ---
def train_one_phase(model, train_loader, num_epochs, lr, criterion, device, description="Phase"):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print(f"{YELLOW}🚀 {description} training for {num_epochs} epochs...{ENDC}")
    start_time = time.time()

    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"{YELLOW}✅ {description} complete. Time: {total_time/60:.1f} min{ENDC}")

    torch.save(model.state_dict(), "mobilenetv2_binary.pth")
    print(f"{YELLOW}📦 Saved model to mobilenetv2_binary.pth{ENDC}")
    return model


# --- שלבי אימון ---
def train_model(num_epochs=10, lr=0.001):
    train_loader, _, _, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    return train_one_phase(model, train_loader, num_epochs, lr, criterion, device, "Basic Training")


def finetune_model(num_epochs=5, lr=0.0001):
    train_loader, _, _, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["features.16", "features.17", "classifier"]):
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    return train_one_phase(model, train_loader, num_epochs, lr, criterion, device, "Shallow Fine-tuning")


def deep_finetune(num_epochs=10, lr=0.0001):
    train_loader, _, _, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(layer in name for layer in [
            "features.8","features.9","features.10","features.11","features.12",
            "features.13","features.14","features.15","features.16","features.17","classifier"
        ]):
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    return train_one_phase(model, train_loader, num_epochs, lr, criterion, device, "Deep Fine-tuning")


def finetune_feature18(num_epochs=5, lr=0.00005):
    train_loader, _, _, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["features.18", "classifier"]):
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    return train_one_phase(model, train_loader, num_epochs, lr, criterion, device, "Feature.18 Fine-tuning")


def weighted_finetune(num_epochs=8, lr=0.0001, weight_factor=1.3):
    train_loader, _, _, _ = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(layer in name for layer in [
            "features.8","features.9","features.10","features.11","features.12",
            "features.13","features.14","features.15","features.16","features.17","classifier"
        ]):
            param.requires_grad = True

    class_weights = torch.tensor([1.0, weight_factor]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return train_one_phase(model, train_loader, num_epochs, lr, criterion, device, "Weighted Fine-tuning")


# --- קוונטיזציה ---
def quantize_and_save(device):
    model = load_model(device)
    model.eval()

    model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)

    # שומרים state_dict נקי (ולא את כל האובייקט!)
    torch.save(model.state_dict(), "../mobilenetv2_NOT_quantized.pth")
    print(f"{YELLOW}📦 Quantized model saved to mobilenetv2_NOT_quantized.pth{ENDC}")
    return model


# --- MAIN PIPELINE ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_model()
    model = finetune_model()
    model = deep_finetune()
    model = finetune_feature18()
    model = weighted_finetune()

    quantize_and_save(device)

    print(f"{YELLOW}🎉 Full training + quantization pipeline finished!{ENDC}")
