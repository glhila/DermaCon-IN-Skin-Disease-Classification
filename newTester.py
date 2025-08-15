import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from data_preparation import prepare_data

# QAT/FX imports (needed for loading QAT checkpoints or rebuilding the INT8 graph)
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
    # Use the same quantize_input flag you used for training; for QAT/INT8 keep this False
    _, _, test_loader, _ = prepare_data(num_workers=0, quantize_input=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===========================
    # Choose what you want to test
    # ===========================
    # 1) FP32 fine-tuned weights (state_dict)
    # model_path = "mobilenetv2_binary.pth"; mode = "fp32"

    # 2) QAT checkpoint (state_dict) — float weights + fake-quant, BEFORE convert_fx
    # model_path = "mobilenetv2_qat_state.pth"; mode = "qat_state"

    # 3) Converted INT8 full module (.pt) — your trusted file
    #    (recommended for deployment/testing)
    # model_path = "mobilenetv2_int8.pt"; mode = "int8_module"

    # 4) Converted INT8 state_dict (.pth) — rebuild INT8 graph then load weights
    # model_path = "mobilenetv2_int8.pth"; mode = "int8_state"

    # 5) (Optional) PT Dynamic quantized Linear-only — not QAT
    # model_path = "mobilenetv2_binary_quantized.pth"; mode = "dynamic_linear"

    # >>> Set yours here <<<
    model_path = "mobilenetv2_int8.pth"
    mode = "int8_state"

    print(f"Mode: {mode} | Model path: {model_path}")

    if mode == "fp32":
        model = build_fp32_model()
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model = model.to(device).eval()

    elif mode == "qat_state":
        # Rebuild float model, prepare for QAT (same backend + example input), THEN load state
        torch.backends.quantized.engine = "fbgemm"  # use "qnnpack" if you trained on ARM/mobile
        model = build_fp32_model().to(device).train()
        example_input = torch.randn(1, 3, 224, 224).to(device)  # must match training size
        qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd)
        model = model.to(device).eval()  # evaluate the QAT graph (fake-quant still active)

    elif mode == "int8_module":
        # Load the fully converted INT8 module (you said the file is trusted)
        # PyTorch 2.6 defaults to weights_only=True; we override since we trust this file.
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()

        # If you prefer to keep weights_only=True, uncomment the allow-list below:
        # from torch.serialization import add_safe_globals
        # from torch.fx.graph_module import reduce_graph_module
        # add_safe_globals([reduce_graph_module])
        # model = torch.load(model_path, map_location=device, weights_only=True)
        # model.eval()

    elif mode == "int8_state":
        # Rebuild the same INT8 structure via FX, then load the INT8 state_dict (no pickle)
        torch.backends.quantized.engine = "fbgemm"
        model = build_fp32_model().to(device).train()
        example_input = torch.randn(1, 3, 224, 224).to(device)  # must match training size
        qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
        model = convert_fx(model).eval()
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)

    elif mode == "dynamic_linear":
        # Post-training dynamic quantization of Linear layers only (not QAT)
        model = build_fp32_model()
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model = model.to(device).eval()

    else:
        raise ValueError("Unknown mode")

    # Evaluate on test set
    evaluate(model, test_loader, device, name="Test")
