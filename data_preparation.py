import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np
import os
import io
from PIL import Image  # For image handling
import torch  # Import torch for PyTorch Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader

# Lightweight wrapper around a DataFrame-backed image store for PyTorch training
class DermaDataset(Dataset):
    def __init__(self, dataframe, transform=None, quantize_input=False, quant_bits=8):
        """
        PyTorch dataset for DermaCon-IN images stored in a DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame with image metadata and an 'image' column containing
                a dict with 'bytes' or 'array'.
            transform (albumentations.Compose, optional): Albumentations transform to apply.
            quantize_input (bool, optional): If True, simulate input quantization before normalization.
            quant_bits (int, optional): Bit depth used when quantizing inputs.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.quantize_input = quantize_input
        self.quant_bits = quant_bits
        self.max_val = 2 ** quant_bits - 1

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve a transformed image tensor and its class label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (image_tensor [C,H,W] float32, label tensor [long])
        """
        row = self.dataframe.iloc[idx]

        image_dict = row['image']

        # Load image from bytes or array and ensure RGB format
        if 'bytes' in image_dict:
            image_bytes = image_dict['bytes']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif 'array' in image_dict:
            image = Image.fromarray(image_dict['array']).convert('RGB')
        else:
            raise ValueError(
                f"Image data not found in 'bytes' or 'array' key for index {idx}. Keys found: {image_dict.keys()}")

        image_np = np.array(image)

        label = row['label']

        if self.transform:
            transformed = self.transform(image=image_np)
            image_np = transformed['image']
            # Note: transforms include resizing and normalization (ImageNet stats)

        # Convert HWC (numpy) to CHW (torch) float32
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        if self.quantize_input:
            # Simulate uniform quantization in [0,1] then renormalize
            image_tensor = torch.clamp(image_tensor, 0, 1)
            image_quantized = (image_tensor * self.max_val).round().byte()
            image_tensor = image_quantized.float() / self.max_val

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            # Rationale: emulate reduced precision input pipeline while keeping model trained on normalized inputs

        return image_tensor, torch.tensor(label, dtype=torch.long)


def prepare_data(batch_size: int = 32, num_workers: int = None,  quantize_input=False):
    """
    Prepare DermaCon-IN data: load, filter, encode, split, transform, and build loaders.

    Args:
        batch_size (int): The number of samples per batch in the DataLoaders. Defaults to 32.
        num_workers (int, optional): The number of subprocesses to use for data loading.
                                     If None, it defaults to half of the available CPU cores.
                                     Set to 0 for single-process loading (useful for debugging).
        quantize_input (bool): If True, enable input quantization in `DermaDataset`.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            - label_encoder (sklearn.preprocessing.LabelEncoder): The encoder used to map labels.
    """
    pass

    try:
        dataset = load_dataset("ekacare/DermaCon-IN", split="train")
        df = dataset.to_pandas()
    except Exception as e:
        raise

    # Focus the task to two super-classes to create a balanced binary classification problem
    target_main_classes = ["Infectious Disorders", "Inflammatory Disorders"]
    df_filtered = df[df["main_class"].isin(target_main_classes)].copy()

    # Encode labels and split
    label_encoder = LabelEncoder()
    df_filtered['label'] = label_encoder.fit_transform(df_filtered['main_class'])

    # Define splitting ratios
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    df_train, df_temp = train_test_split(
        df_filtered,
        test_size=(validation_ratio + test_ratio),
        random_state=42,
        stratify=df_filtered['label']
    )

    df_validation, df_test = train_test_split(
        df_temp,
        test_size=(test_ratio / (validation_ratio + test_ratio)),
        random_state=42,
        stratify=df_temp['label']
    )
    # Stratification preserves label balance across splits

    # Define transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, p=0.7),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    # 224x224 matches common CNN backbones; augmentations regularize; normalization aligns with ImageNet pretraining

    val_test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Create datasets and loaders
    train_dataset = DermaDataset(df_train, transform=train_transform, quantize_input=quantize_input)
    val_dataset = DermaDataset(df_validation, transform=val_test_transform, quantize_input=quantize_input)
    test_dataset = DermaDataset(df_test, transform=val_test_transform, quantize_input=quantize_input)

    if num_workers is None:
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
        # Conservative default: half the CPU cores typically balances throughput and system load

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, label_encoder


if __name__ == "__main__":
    print("Executing data_preparation.py directly.")
    train_loader, val_loader, test_loader, label_encoder = prepare_data(batch_size=64)
    print("\nAll DataLoaders and LabelEncoder are prepared and ready for use.")
    # This entry-point acts as a quick smoke test for data pipeline setup