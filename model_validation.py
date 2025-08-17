import torch
import torch.nn as nn
from torchvision import models
from data_preparation import prepare_data
from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, dataloader, device, name="Validation"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ {name} Accuracy: {acc:.4f} ({sum([p == l for p, l in zip(all_preds, all_labels)])}/{len(all_labels)} correct)")

    # Classification report: Precision, Recall, F1 per class
    print("\n📋 Detailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Infectious", "Inflammatory"]))

    return acc

if __name__ == "__main__":
    print("🚀 Starting validation...")

    # Load only validation data
    _, val_loader, _, label_encoder = prepare_data(num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Decide which model to load
    model_to_validate = "mobilenetv2_binary.pth"

    # Load trained model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    if model_to_validate == "mobilenetv2_NOT_quantized.pth":
        # Apply dynamic quantization wrapper
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    model.load_state_dict(torch.load(model_to_validate, map_location=device))
    model = model.to(device)

    # Run validation
    evaluate(model, val_loader, device, name="Validation")
