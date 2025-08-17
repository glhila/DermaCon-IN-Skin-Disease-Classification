import torch
import torch.nn as nn
from torchvision import models
from data_preparation import prepare_data
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate(model, dataloader, device, name="Test"):
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
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n✅ {name} Accuracy: {acc:.4f}")
    print(f"📊 {name} Confusion Matrix:")
    print(cm)

    # Classification report: Precision, Recall, F1 per class
    print("\n📋 Detailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Infectious", "Inflammatory"]))

    return acc, cm

if __name__ == "__main__":
    print("🚀 Starting model test...")

    # Load test data only
    _, _, test_loader, label_encoder = prepare_data(num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # decide which model to load
    #model_to_load = "mobilenetv2_binary.pth"
    model_to_load = "mobilenetv2_NOT_quantized.pth"

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    if model_to_load == "mobilenetv2_NOT_quantized.pth":
        # Load quantized model
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    model.load_state_dict(torch.load(model_to_load, map_location=device))
    model = model.to(device)

    # Evaluate on test set
    evaluate(model, test_loader, device, name="Test")
