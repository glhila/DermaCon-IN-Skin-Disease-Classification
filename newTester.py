import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from data_preparation import prepare_data

# QAT/FX imports (only needed for qat_state / int8_state modes)
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx


def evaluate(model, dataloader, device, name="Test"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n✅ {name} Accuracy: {acc:.4f}")
    print(f"📊 {name} Confusion Matrix:\n{cm}")
    print("\n📋 Detailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Infectious", "Inflammatory"]))
    return acc, cm


def build_fp32_model(num_classes=2):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


if __name__ == "__main__":
    print("🚀 Starting model test...")

    # --- Match the data pipeline you used for training ---
    # You trained with quantized inputs (NQ), so keep this True here:
    _, _, test_loader, _ = prepare_data(num_workers=0, quantize_input=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===========================
    # Choose what you want to test
    # ===========================
    # 1) FP32 fine-tuned weights (state_dict)
    # model_path = "mobilenetv2_binary.pth"; mode = "fp32"

    # 2) FP32 fine-tuned weights with quantized-input training (your new NQ checkpoint)
    model_path = "mobilenetv2_binary_NQ.pth"; mode = "fp32"

    # 3) QAT checkpoint (state_dict) — BEFORE convert_fx
    # model_path = "mobilenetv2_qat_state.pth"; mode = "qat_state"

    # 4) Converted INT8 full module (.pt)
    # model_path = "mobilenetv2_int8.pt"; mode = "int8_module"

    # 5) Converted INT8 state_dict (.pth) — rebuild INT8 graph then load
    # model_path = "mobilenetv2_int8.pth"; mode = "int8_state"

    # 6) Post-training dynamic quant (Linear-only) — not for your NQ run
    # model_path = "mobilenetv2_binary_quantized.pth"; mode = "dynamic_linear"

    print(f"Mode: {mode} | Model path: {model_path}")

    if mode == "fp32":
        model = build_fp32_model()
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model = model.to(device).eval()

    elif mode == "qat_state":
        torch.backends.quantized.engine = "fbgemm"
        model = build_fp32_model().to(device).train()
        example_input = torch.randn(1, 3, 224, 224).to(device)
        qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd)
        model = model.to(device).eval()

    elif mode == "int8_module":
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()

    elif mode == "int8_state":
        torch.backends.quantized.engine = "fbgemm"
        model = build_fp32_model().to(device).train()
        example_input = torch.randn(1, 3, 224, 224).to(device)
        qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
        model = convert_fx(model).eval()
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)

    elif mode == "dynamic_linear":
        model = build_fp32_model()
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model = model.to(device).eval()

    else:
        raise ValueError("Unknown mode")

    evaluate(model, test_loader, device, name="Test")
