import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader
from training_STE import convert_mobilenetv2_to_bnn

#from data_preparation import prepare_data

# QAT/FX imports (only needed for qat_state / int8_state modes)
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx


def load_model_for_eval(model_path: str, mode: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Load a model given a saved artifact and mode.


Supported modes:
- "fp32": state_dict of a standard FP32 MobileNetV2 classifier
- "bnn_state": state_dict of a MobileNetV2 converted to BNN (binary convs)
- "qat_state": QAT state_dict before convert_fx
- "int8_module": a fully converted/int8 serialized module (.pt)
- "int8_state": converted INT8 state_dict to be loaded into a rebuilt quantized graph
- "dynamic_linear": dynamic quantized Linear-only model with FP32 weights


Notes
-----
• For "qat_state" and "int8_state" we must re-create the *exact* FX graph
(via prepare_qat_fx) with the same input shape used during training.
• "int8_module" expects you saved the *entire module* (e.g., torch.save(model)).
• "bnn_state" requires your convert_mobilenetv2_to_bnn() to match training.
• "dynamic_linear" is post-training dynamic quant (Linear-only) — good baseline.
"""
    if mode == "fp32":
        model = build_fp32_model(num_classes=num_classes)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        return model.to(device).eval()

    if mode == "bnn_state":
        # BNN checkpoint from training_STE.py (binary convs, float classifier)
        model = build_bnn_model(num_classes=num_classes)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        return model.to(device).eval()

    if mode == "qat_state":
        # Rebuild a *QAT-ready* graph then load the QAT state_dict
        torch.backends.quantized.engine = "fbgemm" # x86; use "qnnpack" on ARM/mobile
        model = build_fp32_model(num_classes=num_classes).to(device).train()
        example_input = torch.randn(1, 3, 224, 224).to(device) # must match training input size
        qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
        sd = torch.load(model_path, map_location="cpu") # safer when keys live on CPU
        model.load_state_dict(sd)
        return model.to(device).eval()

    if mode == "int8_module":
        # Load a fully converted INT8 module (not just state_dict)
        model = torch.load(model_path, map_location=device, weights_only=False)
        return model.eval()

    if mode == "int8_state":

        torch.backends.quantized.engine = "fbgemm"
        model = build_fp32_model(num_classes=num_classes).to(device).train()
        example_input = torch.randn(1, 3, 224, 224).to(device)
        qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs=example_input)
        model = convert_fx(model).eval()
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        return model

    if mode == "dynamic_linear":
        # Post-training dynamic quantization for Linear layers only (no Conv quant)
        model = build_fp32_model(num_classes=num_classes)
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        return model.to(device).eval()

    raise ValueError("Unknown mode")


def evaluate_saved_model(model_path: str, mode: str = "fp32", name: str = "Test", test_loader :DataLoader = None) -> None:
    """"High-level helper: (optionally) build test loader, load model from path, and evaluate.


    Parameters
    ----------
    model_path : str
    Path to weights/module to load.
    mode : str
    One of {"fp32", "bnn_state", "qat_state", "int8_module", "int8_state", "dynamic_linear"}.
    name : str
    Label for printed metrics.
    test_loader : DataLoader | None
    Dataloader yielding (images, labels). If None, this function will raise — supply your own
    loader or plug your data prep here.
    class_names : list[str] | None
    Names for classes in the classification report. If None, generic numeric labels are used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_eval(model_path, mode, device)
    evaluate(model, test_loader, device, name=name)


def evaluate(model, dataloader, device, name="Test"):
    """Run inference over a dataloader and print accuracy, confusion matrix, and a report.


    Notes
    -----
    • Uses torch.inference_mode() for speed and to avoid autograd overhead.
    • Accumulates predictions/labels on CPU lists to control memory.
    • classification_report supports label names if you pass class_names.
    """
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
    print(f"\n {name} Accuracy: {acc:.4f}")
    print(f" {name} Confusion Matrix:\n{cm}")
    print("\n Detailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Infectious", "Inflammatory"]))
    return acc, cm


def build_fp32_model(num_classes=2):
    """Construct a MobileNetV2 backbone with a custom classifier head.


    NOTE: We use weights=None because we typically load a state_dict. If you want
    to evaluate ImageNet-pretrained features without loading a checkpoint, switch to
    weights=models.MobileNet_V2_Weights.DEFAULT.
    """
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

def build_bnn_model(num_classes=2, keep_first_conv_fp=True):
    # same base arch as training: MobileNetV2 with 2-class head
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    # convert convs to binarized layers exactly like training
    m = convert_mobilenetv2_to_bnn(
        m,
        keep_first_conv_fp=keep_first_conv_fp,  # True in your training
        quant_mode='det',
        allow_scale=False
    )
    return m



if __name__ == "__main__":
    print(" Starting model test...")

