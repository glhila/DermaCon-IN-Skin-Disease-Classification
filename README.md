# Skin Disease Classification

## Overview

This project is an advanced computer architectures course assignment: a deep‑learning pipeline for classifying infectious and inflammatory skin diseases in dark‑skinned individuals. It provides Python code to prepare data, train MobileNetV2, and evaluate saved models, supporting both standard (non‑quantized) training and quantization (QAT/STE) with PyTorch.

## Project Structure

```
Skin-Disease-Classification/
│
├── src/                              # Source code
│   ├── data_preparation.py           # Builds datasets, transforms, and DataLoaders
│   ├── main.py                       # Main entry point to run training and evaluation
│   ├── model_validation.py           # Utilities for model evaluation and metrics
│   ├── training_STE.py           # Quantized training (QAT/STE) for MobileNetV2
│   ├── Tester.py                  # Standalone evaluation of saved checkpoints
│   └── train_model_NQ.py             # Non‑quantized training loop (MobileNetV2 backbone)
│
├── results/                          # Directory for saving trained model checkpoints
│   ├── mobilenetv2_best_not_quantized.pth       # Non-quantized data + Non-quantized model (Mode A)
│   ├── mobilenetv2_best_data_quantized.pth      # Quantized data + Non-quantized model (Mode B)
│   ├── mobilenetv2_best_model_quantized.pth     # Non-quantized data + Quantized model (Mode C)
│   └── mobilenetv2_best_fully_quantized.pth     # Quantized data + Quantized model (Mode D)
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation (this file)

```
---

## Getting Started

### Prerequisites

* **Python:** Ensure you have Python 3.10 or higher installed.
* **Git:** You need Git to clone the repository.

### Installation

1.  **Clone the repository:**
    ```sh
    cd Skin-Disease-Classification
    ```
2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage


### 1. Configure the Run Mode

The `main.py` file allows you to select one of four training and evaluation modes. Open `main.py` and modify the `__main__` block to call `run_models()` with the desired mode.

* **'A'**: Non-quantized data + Non-quantized model
* **'B'**: Quantized data + Non-quantized model
* **'C'**: Non-quantized data + Quantized model
* **'D'**: Quantized data + Quantized model

Example for running mode 'A':
```python
if __name__ == '__main__':
    run_models('A')
```

### 2. Run the Script

Once you have configured the desired mode in `main.py`, simply run the script from your terminal:
```sh
python main.py
```
The script will handle the entire process: preparing the data, training the model according to the selected mode, and evaluating its performance.

---

## Troubleshooting

* **PyTorch/CUDA mismatch:** If you encounter issues, ensure your PyTorch build matches your installed CUDA toolkit version.
* **Dataset path errors:** Verify that the dataset paths specified in `data_preparation.py` are correct.
* **Dependency errors:** Confirm that all dependencies were installed correctly by running `pip install -r requirements.txt`.

---

## Project Documentation

* **Project Documentation:** [View the project documentation here](<https://docs.google.com/document/d/1h9M2Dn-k3A0Fy51yO-_n6odgEBsHchEi3rhDAGrPjsE/edit?usp=sharing>)
* **Project Presentation:** [View the project presentation here]()
