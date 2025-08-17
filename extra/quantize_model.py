import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def quantize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. Load trained model ===
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

    mobilenet.load_state_dict(torch.load("../old_training/mobilenetv2_binary.pth", map_location=device))
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    print("✅ Original model loaded")

    # === 2. Apply dynamic quantization ===
    quantized_model = torch.quantization.quantize_dynamic(
        mobilenet, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "../mobilenetv2_NOT_quantized.pth")
    print("✅ Quantized model saved")

if __name__ == "__main__":
    quantize()
