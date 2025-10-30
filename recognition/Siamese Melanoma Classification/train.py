"""
train.py
---------
Handles model training, validation, and testing for the Siamese network.
Includes accuracy evaluation, metric tracking, and result plotting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from modules import SiameseNetwork
from dataset import SiameseMelanomaDataset  # assuming your dataset is in dataset.py

class SiameseTrainer:
    """
    Trainer class for SiameseNetwork on melanoma dataset.

    Handles dataset creation, dataloaders, weight initialization, and contrastive loss.
    """
    def __init__(self, train_benign, train_malignant, test_benign, test_malignant,
                 transform=None, embedding_dim=256, freeze_base=True, fine_tune_from_block=2,
                 batch_size=16, device=None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(42)
        self.batch_size = batch_size

        # Initialize model
        self.model = SiameseNetwork(freeze_base=freeze_base,
                                    fine_tune_from_block=fine_tune_from_block,
                                    embedding_dim=embedding_dim).to(self.device)
        self._initialize_weights()

        # Create datasets
        self.train_dataset = SiameseMelanomaDataset(train_benign, train_malignant,
                                                    transform=transform, num_pairs=10000)
        self.test_dataset = SiameseMelanomaDataset(test_benign, test_malignant,
                                                   transform=transform, num_pairs=2000)

        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def _initialize_weights(self):
        for name, param in self.model.named_parameters():
            if "weight" in name:
                self._W_init_torch(param)
            elif "bias" in name:
                self._b_init_torch(param)

    def _W_init_torch(self, tensor):
        """Initialize weights as N(0, 0.01)"""
        with torch.no_grad():
            values = self.rng.normal(loc=0, scale=1e-2, size=tensor.shape).astype(np.float32)
            tensor.copy_(torch.from_numpy(values))

    def _b_init_torch(self, tensor):
        """Initialize biases as N(0.5, 0.01)"""
        with torch.no_grad():
            values = self.rng.normal(loc=0.5, scale=1e-2, size=tensor.shape).astype(np.float32)
            tensor.copy_(torch.from_numpy(values))

    def contrastive_loss(self, output1, output2, label, margin=1.0):
        """
        Computes the contrastive loss between pairs of embeddings.
        """
        dist = F.pairwise_distance(output1, output2, p=2)
        loss = torch.mean((1 - label) * torch.pow(dist, 2) +
                          label * torch.pow(torch.clamp(margin - dist, min=0.0), 2))
        return loss

    def forward(self, x1, x2):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        return self.model(x1, x2)

'''trainer = SiameseTrainer(train_benign, train_malignant,
                         test_benign, test_malignant,
                         transform=transform,
                         batch_size=16)

# Quick test
for img1, img2, label in trainer.train_loader:
    out = trainer.forward(img1, img2)
    print(out.shape)  # should be [batch_size, 1]
    break
'''