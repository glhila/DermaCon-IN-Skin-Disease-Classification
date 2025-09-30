"""
Run four experiment configurations:


3 Baselines
-----------
A: Non-quantized data + Non-quantized model (FP32)
B: Quantized data + Non-quantized model (FP32 + quantized inputs)
C: Non-quantized data + Quantized model (BNN/QAT/STE style weights)


1 Final Model
-------------
D: Quantized data + Quantized model (BNN/QAT/STE + quantized inputs)


Notes
-----
• Assumes the following modules provide the specified functions:
- data_preparation.prepare_data(batch_size, num_workers, quantize_input) -> (train, val, test, meta)
- train_model_NQ.train_model(...): FP32 fine-tuning (saves mobilenetv2_best_not_quantized.pth or
mobilenetv2_best_data_quantized.pth depending on data_is_quantized)
- New_training_STE.train_model_quantized(...): BNN/QAT training (saves *_model_quantized.pth / *_fully_quantized.pth)
- newTester.evaluate_saved_model(path, mode, test_loader=...): evaluation utility


• If you change checkpoint names in training functions, update the evaluate paths here.
• Consider seeding RNGs (torch, random, numpy) in prepare_data or here for reproducibility.
"""

from data_preparation import prepare_data
from train_model_NQ import train_model
from training_STE import train_model_quantized
from model_tester import evaluate_saved_model

def run_models(mode):
    match mode:
        case 'A':
            # Baseline A: Standard FP32 training on normal (non-quantized) inputs
            print("Running Model A: Non-quantized data + Non-quantized model")
            # Prepare dataloaders with raw (non-quantized) data
            train_loader, val_loader, test_loader, _ = prepare_data(batch_size=32, num_workers=0, quantize_input=False)
            # Train standard FP32 model
            train_model(data_is_quantized=False, stage_epochs=(3,10,12), early_stop_patience=5, train_loader=train_loader, val_loader=val_loader)
            # Evaluate using saved FP32 checkpoint
            evaluate_saved_model("mobilenetv2_best_not_quantized.pth", mode="fp32", test_loader=test_loader)

        case 'B':
            # Baseline B: FP32 model trained with quantized input data
            print("Running Model B: Quantized data + Non-quantized model")
            # Prepare dataloaders with quantized inputs
            train_loader, val_loader, test_loader, _ = prepare_data(batch_size=32, num_workers=0, quantize_input=True)
            # Train standard FP32 model with quantized inputs
            train_model(data_is_quantized=True, stage_epochs=(6,20,24), early_stop_patience=6, train_loader=train_loader, val_loader=val_loader)
            # Evaluate using saved FP32 checkpoint
            evaluate_saved_model("mobilenetv2_best_data_quantized.pth", mode="fp32", test_loader=test_loader)
        
        case 'C':
            # Model C: BNN/QAT model trained with non-quantized input data
            print("Running Model C: Non-quantized data + Quantized model")
            # Prepare dataloaders with raw (non-quantized) data
            train_loader, val_loader, test_loader, _ = prepare_data(batch_size=32, num_workers=0, quantize_input=False)
            # Train BNN/QAT model with non-quantized inputs
            train_model_quantized(data_is_quantized=False, stage_epochs=(3,10,12), early_stop_patience=5, train_loader=train_loader, val_loader=val_loader)
            # Evaluate using saved BNN/QAT checkpoint
            evaluate_saved_model("mobilenetv2_best_model_quantized.pth", mode="bnn_state", test_loader=test_loader)
        
        case 'D':
            # Final Model D: BNN/QAT model trained with quantized input data
            print("Running Model D: Quantized data + Quantized model")
            # Prepare dataloaders with quantized inputs
            train_loader, val_loader, test_loader, _ = prepare_data(batch_size=32, num_workers=0, quantize_input=True)
            # Train BNN/QAT model with quantized inputs
            train_model_quantized(data_is_quantized=True, stage_epochs=(6,20,24), early_stop_patience=6, train_loader=train_loader, val_loader=val_loader)
            # Evaluate using saved fully quantized checkpoint
            evaluate_saved_model("mobilenetv2_best_fully_quantized.pth", mode="bnn_state", test_loader=test_loader)
        
        case _:
            # Invalid option handler
            raise ValueError("Invalid mode. Choose 'A' or 'B' or 'C' or 'D'.")

if __name__ == "__main__":
    # Uncomment the desired run to execute specific experiment
    #run_models('A')
    #run_models('B')
    run_models('C')
    #run_models('D')