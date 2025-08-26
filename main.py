# 3 Baseline models:
# A: Non-quantized data + Non-quantized model
# B: Quantized data     + Non-quantized model
# C: Non-quantized data + Quantized model
# ------------------------------------------------------------
# 1 Final model:
# D: Quantized data     + Quantized model
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data
from train_model_NQ import train_model
from newTester import evaluate_saved_model

def run_models(mode):
    match mode:
        case 'A':
            print("Running Model A: Non-quantized data + Non-quantized model")
            train_loader, val_loader, test_loader, _ = prepare_data(batch_size=32, num_workers=0, quantize_input=False)
            train_model(data_is_quantized=False, stage_epochs=(3,10,12), early_stop_patience=5, train_loader=train_loader, val_loader=val_loader)
            evaluate_saved_model("mobilenetv2_best_not_quantized.pth", mode="fp32", test_loader=test_loader)

        case 'B':
            print("Running Model B: Quantized data + Non-quantized model")
            train_loader, val_loader, test_loader, _ = prepare_data(batch_size=32, num_workers=0, quantize_input=True)
            train_model(data_is_quantized=True, stage_epochs=(6,20,24), early_stop_patience=6, train_loader=train_loader, val_loader=val_loader)
            evaluate_saved_model("mobilenetv2_best_data_quantized.pth", mode="fp32", test_loader=test_loader)

        case _:
            raise ValueError("Invalid mode. Choose 'A' or 'B' or 'C' or 'D'.")

    #elif mode == 'C':
    #    print("Running Model C: Non-quantized data + Quantized model")
    #    train_model(data_is_quantized=False, model_is_quantized=True)
    #elif mode == 'D':
    #    print("Running Model D: Quantized data + Quantized model")
    #    train_model(data_is_quantized=True, model_is_quantized=True)
    #else:
    #    print("Invalid mode. Please choose from 'A', 'B', 'C', or 'D'.")

if __name__ == "__main__":

    run_models('A')
    #run_models('B')
    #run_models('C')
    #run_models('D')