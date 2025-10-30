"""
dataset.py
-----------
Defines the custom SiameseMelanomaDataset class for generating
image pairs (benign vs malignant) and performing augmentations.
"""
import torch

from torchvision import transforms

import numpy as np

from torch.utils.data import Dataset

from PIL import Image

IMAGE_DIR = "/kaggle/input/isic-2020-jpg-224x224-resized/train-image/image"
TRAIN_CSV_PATH = "/kaggle/input/isic-2020-jpg-224x224-resized/train-metadata.csv"

IMG_SIZE = 224

# transformations on the images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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
        self.transform = transform
        self.benign_paths = benign_paths
        self.malignant_paths = malignant_paths
        self.num_pairs = num_pairs
        self.rng = np.random.default_rng(42)

        # Upsample malignant images to match number of benign pairs
        self.malignant_weight = len(benign_paths) / len(malignant_paths)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 50% similar, 50% dissimilar
        same_class = self.rng.random() < 0.5

        if same_class:
            if self.rng.random() < 0.5:
                # benign
                img1_path, img2_path = self.rng.choice(self.benign_paths, size=2, replace=False)
                label = 0
            else:
                # malignant (upsample with replacement)
                img1_path, img2_path = self.rng.choice(self.malignant_paths, size=2, replace=True)
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
