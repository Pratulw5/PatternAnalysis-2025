import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
class TripletMelanomaDataset(Dataset):
    """
    Dataset that creates triplets of images for Siamese network training.
    Each sample contains: anchor, positive (same class), negative (different class)
    """
    
    def __init__(self, benign_paths, malignant_paths, transform=None, 
                 num_triplets=10000, seed=None):
        """
        Args:
            benign_paths (list): List of file paths to benign images
            malignant_paths (list): List of file paths to malignant images
            transform: Torchvision transforms to apply
            num_triplets (int): Number of triplets to generate
            seed (int): Random seed for reproducibility
        """
        self.transform = transform
        self.benign_paths = benign_paths
        self.malignant_paths = malignant_paths
        self.num_triplets = num_triplets
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate triplets at initialization
        self.triplets = []
        self._generate_triplets()
    
    def _generate_triplets(self):
        """
        Pre-generate all triplets once at initialization.
        Ensures consistent dataset across epochs.
        """
        for _ in range(self.num_triplets):
            # 50% chance anchor is benign, 50% malignant
            if self.rng.random() < 0.5:
                # Anchor and positive are benign, negative is malignant
                anchor, positive = self.rng.choice(
                    self.benign_paths, size=2, replace=False
                )
                negative = self.rng.choice(self.malignant_paths)
                label = 0  # benign
            else:
                # Anchor and positive are malignant, negative is benign
                anchor, positive = self.rng.choice(
                    self.malignant_paths, size=2, replace=True
                )
                negative = self.rng.choice(self.benign_paths)
                label = 1  # malignant
            
            self.triplets.append((anchor, positive, negative, label))
    
    def __len__(self):
        return self.num_triplets
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (anchor, positive, negative, label)
        """
        anchor_path, positive_path, negative_path, label = self.triplets[idx]
        
        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative, torch.tensor(label, dtype=torch.long)



def get_transforms(img_size=224, mode='train'):
    """
        Get appropriate transforms for train/test
        
        Args:
            img_size (int): Size to resize images to
            mode (str): 'train' or 'test'
            
        Returns:
            torchvision.transforms.Compose: Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    else:  # test mode
        return transforms.Compose([transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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