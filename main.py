# file: four_models_baseline.py
# ------------------------------------------------------------
# Baseline for 4 models:
# A: Non-quantized data + Non-quantized model
# B: Quantized data     + Non-quantized model
# C: Non-quantized data + Quantized model
# D: Quantized data     + Quantized model
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from data_preparation import prepare_data
import train_model_NQ
from train_model_NQ import train_model

def run_models(mode):
    if mode == 'A':
        print("Running Model A: Non-quantized data + Non-quantized model")
        train_model(data_is_quantized=False)
    elif mode == 'B':
        print("Running Model B: Quantized data + Non-quantized model")
        train_model(data_is_quantized=True)

    #elif mode == 'C':
    #    print("Running Model C: Non-quantized data + Quantized model")
    #    train_model(data_is_quantized=False, model_is_quantized=True)
    #elif mode == 'D':
    #    print("Running Model D: Quantized data + Quantized model")
    #    train_model(data_is_quantized=True, model_is_quantized=True)
    #else:
    #    print("Invalid mode. Please choose from 'A', 'B', 'C', or 'D'.")

if __name__ == "__main__":
    #run_models('A')
    run_models('B')
    #run_models('C')
    #run_models('D')