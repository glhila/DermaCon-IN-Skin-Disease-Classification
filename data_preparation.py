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


# --- Custom PyTorch Dataset Class ---
# This class defines how our dataset interacts with PyTorch.
# It's placed here (globally) so it can be accessed by the prepare_data function.
class DermaDataset(Dataset):
    def __init__(self, dataframe, transform=None, quantize_input=False):
        """
        Initializes the DermaDataset.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame containing image metadata and the 'image' column.
                                      The 'image' column is expected to hold a dictionary with 'bytes' or 'array' keys.
            transform (albumentations.Compose, optional): The image transformation pipeline. Defaults to None.
            quantize_input (bool, optional): Whether to quantize the input images. Defaults to False.
            quant_bits (int, optional): Number of bits for quantization. Defaults to 8.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.quantize_input = quantize_input

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image tensor and its label tensor.
        """
        row = self.dataframe.iloc[idx]

        # Access the 'image' dictionary from the DataFrame row.
        image_dict = row['image']

        # Load image data from bytes or a pre-existing array within the dictionary.
        # We convert to 'RGB' to ensure consistent channel order.
        if 'bytes' in image_dict:
            image_bytes = image_dict['bytes']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif 'array' in image_dict:
            image = Image.fromarray(image_dict['array']).convert('RGB')
        else:
            # Raise an error if image data isn't in the expected 'bytes' or 'array' format.
            raise ValueError(
                f"Image data not found in 'bytes' or 'array' key for index {idx}. Keys found: {image_dict.keys()}")

        # Convert the PIL Image to a NumPy array, which Albumentations expects.
        image_np = np.array(image)

        # Retrieve the numerical label for the current sample.
        label = row['label']

        # Apply transformations if a transform pipeline is provided.
        if self.transform:
            # Albumentations expects a NumPy array (H, W, C).
            transformed = self.transform(image=image_np)
            image_np = transformed['image']  # Get the transformed image from the dictionary output.

        # Convert the processed NumPy array to a PyTorch tensor.
        # Albumentations outputs HWC (Height, Width, Channels), but PyTorch expects CHW.
        # .float() converts the tensor to float type, necessary for model input.
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        if self.quantize_input:
            # Quantization process
            image_tensor = torch.clamp(image_tensor, 0, 1)  # Ensure [0,1] range
            threshold = 0.5
            image_binary = (image_tensor > threshold).float()
            #image_quantized = (image_tensor * self.max_val).round().byte()
            #image_tensor = image_quantized.float() / self.max_val

            # Reapply normalization (important!)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_binary - mean) / std

        # Convert the label to a PyTorch tensor with Long data type (common for classification targets).
        return image_tensor, torch.tensor(label, dtype=torch.long)


# --- Main Data Preparation Function ---
# This function encapsulates all steps required to prepare the dataset.
def prepare_data(batch_size: int = 32, num_workers: int = None, quantize_input=False):
    """
    Prepares the dermatological image data for deep learning training.

    This function orchestrates the following key steps:
    1. Loads and filters the raw DermaCon-IN dataset.
    2. Encodes the 'main_class' categories into numerical labels.
    3. Splits the data into training, validation, and test sets, ensuring label distribution is stratified.
    4. Defines image augmentation and normalization pipelines using Albumentations.
    5. Creates PyTorch Dataset and DataLoader instances for efficient data batching during model training.

    Args:
        batch_size (int): The number of samples per batch in the DataLoaders. Defaults to 32.
        num_workers (int, optional): The number of subprocesses to use for data loading.
                                     If None, it defaults to half of the available CPU cores.
                                     Set to 0 for single-process loading (useful for debugging).

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            - label_encoder (sklearn.preprocessing.LabelEncoder): The encoder used to map labels.
    """
    print("--- Starting Data Preparation Process ---")

    # --- Step 1: Load and Filter the DermaCon-IN dataset ---
    print("Loading the DermaCon-IN dataset...")
    try:
        dataset = load_dataset("ekacare/DermaCon-IN", split="train")
        df = dataset.to_pandas()
        print(f"Dataset loaded successfully. Total samples: {len(df)}")
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        print("Please ensure the 'datasets' library is installed and the dataset is available.")
        # Re-raise the exception to indicate a critical failure in data loading.
        raise

    target_main_classes = ["Infectious Disorders", "Inflammatory Disorders"]
    df_filtered = df[df["main_class"].isin(target_main_classes)].copy()

    print(f"Filtered dataset contains {len(df_filtered)} samples for target classes: {target_main_classes}")
    print("Distribution of main classes in the filtered dataset:")
    print(df_filtered["main_class"].value_counts())

    # --- Step 2: Encode Labels and Split Dataset ---
    print("\nEncoding Labels and Splitting Dataset...")
    label_encoder = LabelEncoder()
    df_filtered['label'] = label_encoder.fit_transform(df_filtered['main_class'])

    # Display the mapping created by the label encoder (e.g., 'Infectious Disorders': 0).
    print("Label mapping:")
    for i, label_name in enumerate(label_encoder.classes_):
        print(f"  {label_name}: {i}")

    # Define splitting ratios
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15

    # Split the dataset into a training set and a temporary set (validation + test).
    df_train, df_temp = train_test_split(
        df_filtered,
        test_size=(validation_ratio + test_ratio),  # Size of the combined temporary set.
        random_state=42,  # Ensures reproducibility of the split.
        stratify=df_filtered['label']  # Maintains the proportion of labels in each split.
    )

    # Split the temporary set into distinct validation and test sets.
    df_validation, df_test = train_test_split(
        df_temp,
        test_size=(test_ratio / (validation_ratio + test_ratio)),  # Proportion of temp set for test.
        random_state=42,  # Ensures reproducibility.
        stratify=df_temp['label']  # Maintains label distribution within temp split.
    )

    # Print the final sizes of each dataset split.
    print(f"Dataset split into:")
    print(f"  Training set size: {len(df_train)}")
    print(f"  Validation set size: {len(df_validation)}")
    print(f"  Test set size: {len(df_test)}")

    # Print the normalized distribution of labels for each split to confirm stratification worked.
    print("\nDistribution of labels across splits (normalized):")
    print(f"  Train: \n{df_train['label'].value_counts(normalize=True)}")
    print(f"  Validation: \n{df_validation['label'].value_counts(normalize=True)}")
    print(f"  Test: \n{df_test['label'].value_counts(normalize=True)}")

    # --- Step 3: Define Image Augmentation Transforms ---
    print("\nDefining Image Augmentation Transforms...")

    if quantize_input:
        # When quantizing inside DermaDataset, skip Albumentations Normalize here
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.GaussNoise(p=0.2),
        ])

        val_test_transform = A.Compose([
            A.Resize(224, 224),
        ])
    else:
        # Original behavior: Normalize with ImageNet stats in Albumentations
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

        val_test_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    print("Image augmentation transforms defined successfully.")

    # --- Step 4: Create Dataset Instances and DataLoaders ---
    print("\nCreating Dataset Instances and DataLoaders...")
    # Create instances of our custom DermaDataset for each data split.
    train_dataset = DermaDataset(df_train, transform=train_transform, quantize_input=quantize_input)
    val_dataset = DermaDataset(df_validation, transform=val_test_transform, quantize_input=quantize_input)
    test_dataset = DermaDataset(df_test, transform=val_test_transform, quantize_input=quantize_input)

    # Determine the number of worker processes for DataLoader.
    if num_workers is None:
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
        if num_workers == 0:
            print("Warning: num_workers set to 0. Data loading will be single-threaded, which might be slow.")

    # Create PyTorch DataLoaders. These will batch and shuffle the data for training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders created with batch_size={batch_size} and num_workers={num_workers}.")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # --- Optional: Verify a batch from DataLoader ---
    # This block tests if the data loading pipeline works correctly by fetching one batch.
    print("\nVerifying a sample batch from train_loader...")
    try:
        # Get one batch (images and labels) from the training data loader.
        for images, labels in train_loader:
            print(f"Shape of image batch: {images.shape}")  # Expected: [batch_size, 3, 224, 224]
            print(f"Shape of label batch: {labels.shape}")  # Expected: [batch_size]
            print(f"First 5 labels: {labels[:5].tolist()}")  # Convert to list for clearer output.
            break  # Stop after getting the first batch.
        print("Sample batch loaded successfully. Data preparation pipeline confirmed.")
    except Exception as e:
        print(f"Error loading sample batch from DataLoader: {e}")
        print("Please check your DermaDataset and DataLoader setup for potential issues.")

    print("\n--- Data Preparation Complete ---")

    # Return the DataLoaders and the label encoder for use in other modules.
    return train_loader, val_loader, test_loader, label_encoder


# --- Main execution block for direct script run ---
# This block runs only when data_preparation.py is executed directly (e.g., `python data_preparation.py`).
# It does NOT run when this file is imported as a module into another script.
if __name__ == "__main__":
    print("Executing data_preparation.py directly.")
    # Example usage: prepare data with a batch size of 64.
    train_loader, val_loader, test_loader, label_encoder = prepare_data(batch_size=64)
    print("\nAll DataLoaders and LabelEncoder are prepared and ready for use.")