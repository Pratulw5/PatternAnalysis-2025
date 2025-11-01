import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
class SiameseMelanomaDataset(Dataset):
    """
    Dataset for Siamese network training with benign/malignant image pairs.

    Each item returns a pair of images and a label (0 = similar, 1 = dissimilar),
    with malignant images upsampled to balance the dataset.

    Args:
        benign_paths (list): Paths to benign images.
        malignant_paths (list): Paths to malignant images.
        transform (callable, optional): Image transformations.
        num_pairs (int): Number of pairs to generate.
    """
    def __init__(self, benign_paths, malignant_paths, transform=None, num_pairs=10000):
        """
        Initializes the SiameseMelanomaDataset.

        Args:
            benign_paths (list): List of paths to benign images.
            malignant_paths (list): List of paths to malignant images.
            transform (callable, optional): Transformations to apply to images.
            num_pairs (int): Number of image pairs to generate.
        """
        self.transform = transform
        self.benign_paths = benign_paths
        self.malignant_paths = malignant_paths
        self.num_pairs = num_pairs
        self.rng = np.random.default_rng(42)

    def __len__(self):
        """
        Returns the number of pairs in the dataset.

        Returns:
            int: Number of image pairs.
        """
        return self.num_pairs

    def __getitem__(self, idx):
        """
        Generates a single image pair and label.

        50% of the time, returns a pair from the same class (similar = 0),
        50% of the time, returns a pair from different classes (dissimilar = 1).
        Malignant images are upsampled as needed.

        Args:
            idx (int): Index of the pair (ignored, pairs are randomly sampled).

        Returns:
            tuple: (img1, img2, label)
                img1 (torch.Tensor): First image tensor.
                img2 (torch.Tensor): Second image tensor.
                label (torch.Tensor): Float tensor (0 = similar, 1 = dissimilar).
        """
        # Decide if the pair is from the same class or not
        same_class = self.rng.random() < 0.5

        if same_class:
            if self.rng.random() < 0.5:
                # Benign pair
                img1_path, img2_path = self.rng.choice(
                    self.benign_paths, size=2, replace=len(self.benign_paths) < 2
                )
            else:
                # Malignant pair (upsample with replacement if needed)
                img1_path, img2_path = self.rng.choice(
                    self.malignant_paths, size=2, replace=len(self.malignant_paths) < 2
                )
            label = 0
        else:
            # Dissimilar pair: one benign, one malignant
            img1_path = self.rng.choice(self.benign_paths)
            img2_path = self.rng.choice(self.malignant_paths)
            label = 1

        # Load images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def get_transforms(img_size=224):
    """
    Returns the default image transformations for training.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
# Data Splitting

def load_data_splits(image_dir, csv_path, sample_size=584, 
                     train_ratio=0.8, seed=42):
    """
    Load and split data into train/test sets with balanced sampling
    
    Args:
        image_dir (str): Directory containing images
        csv_path (str): Path to metadata CSV file
        sample_size (int): Number of samples per class to use
        train_ratio (float): Ratio of data to use for training
        seed (int): Random seed
        
    Returns:
        tuple: (train_benign, train_malignant, test_benign, test_malignant)
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"Total images in metadata: {len(df)}")
    
    # Separate by class
    benign_ids = df[df['target'] == 0]['isic_id'].values
    malignant_ids = df[df['target'] == 1]['isic_id'].values
    
    print(f"Benign: {len(benign_ids)}, Malignant: {len(malignant_ids)}")
    
    # Sample balanced subset
    rng = np.random.default_rng(seed)
    sampled_benign = rng.choice(benign_ids, sample_size, replace=False)
    sampled_malignant = rng.choice(malignant_ids, sample_size, replace=False)
    
    # Convert to full paths
    benign_paths = [
        os.path.join(image_dir, f"{img_id}.jpg") 
        for img_id in sampled_benign
    ]
    malignant_paths = [
        os.path.join(image_dir, f"{img_id}.jpg") 
        for img_id in sampled_malignant
    ]
    
    # Split into train/test
    train_split = int(train_ratio * sample_size)
    
    train_benign = benign_paths[:train_split]
    test_benign = benign_paths[train_split:]
    train_malignant = malignant_paths[:train_split]
    test_malignant = malignant_paths[train_split:]
    
    print(f"\nTrain: {len(train_benign)} benign, {len(train_malignant)} malignant")
    print(f"Test: {len(test_benign)} benign, {len(test_malignant)} malignant")
    
    return train_benign, train_malignant, test_benign, test_malignant