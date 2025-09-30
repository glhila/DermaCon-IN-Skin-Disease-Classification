import torch
import torch.nn as nn
from data_preparation import prepare_data
from sklearn.metrics import classification_report, accuracy_score
from model_testing import load_model_for_eval


def evaluate(model, dataloader, device, name="Validation"):
    """Run inference over a dataloader and compute accuracy, loss, and classification metrics.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader yielding (images, labels)
        device: torch.device for computation
        name: Label for printed metrics

    Returns:
        tuple: (accuracy, average_loss)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Accumulate loss over the entire dataloader
            batch_loss = nn.CrossEntropyLoss()(outputs, labels)
            batch_size = labels.size(0)
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size

    # Average loss over all samples
    avg_loss = total_loss / max(total_samples, 1)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n {name} Accuracy: {acc:.4f} ({sum([p == l for p, l in zip(all_preds, all_labels)])}/{len(all_labels)} correct)")

    # Classification report: Precision, Recall, F1 per class
    print("\n Detailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Infectious", "Inflammatory"]))

    return acc, avg_loss

if __name__ == "__main__":
    print(" Starting validation...")

    # Load only validation data
    _, val_loader, _, _ = prepare_data(num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Choose the model checkpoint and mode to validate.
    # Modes supported (see load_model_for_eval):
    # "fp32", "bnn_state", "qat_state", "int8_module", "int8_state", "dynamic_linear"
    model_path = "mobilenetv2_best_not_quantized.pth"
    mode = "fp32"

    # Build and load the appropriate model for evaluation
    model = load_model_for_eval(model_path=model_path, mode=mode, device=device, num_classes=2)

    # Run validation
    evaluate(model, val_loader, device, name=f"Validation ({mode})")
