import torch
import torch.nn as nn
from torchvision import models
from data_preparation import prepare_data

def evaluate(model, dataloader, device, name="Validation"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = correct / total
    print(f"✅ {name} Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    print("🚀 Starting validation...")
    # טוען את הנתונים
    _, val_loader, _, label_encoder = prepare_data(num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # בונה מחדש את המודל ומטעין את המשקלים השמורים
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("mobilenetv2_binary.pth", map_location=device))
    model.to(device)

    # מריץ ולידציה
    evaluate(model, val_loader, device, name="Validation")
