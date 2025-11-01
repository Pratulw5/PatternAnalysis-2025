"""
modules.py
-----------
Defines all model components for the Siamese Melanoma Detection system.
Includes the Siamese network architecture, embedding layers, and similarity heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    """
    Siamese CNN using EfficientNet-B0 as backbone for contrastive learning.

    Takes a pair of images and produces embeddings. Contrastive loss
    can be computed between embeddings of similar/dissimilar pairs.

    Args:
        base_model_name (str): Backbone model name, currently only "efficientnet_b0".
        embedding_dim (int): Dimension of the embedding vector.
        freeze_base (bool): Freeze early layers of the backbone if True.
        fine_tune_from_block (int): Block index from which to unfreeze for fine-tuning.
    """
    def __init__(self, embedding_dim=256, pretrained=True):
        super(SiameseNetwork, self).__init__()

        # Load EfficientNet-B0 backbone
        if pretrained:
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            base_model = models.efficientnet_b0(weights=None)
        
        self.feature_extractor = nn.Sequential(*list(base_model.features.children()))

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        # Classification head for auxiliary task
        self.classifier = nn.Sequential(nn.ReLU(),nn.Dropout(0.3),nn.Linear(embedding_dim, 2))

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

    def classify(self, x):
        """
        Perform classification on input image
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Classification logits [B, 2]
        """
        embed = self.forward_once(x)
        return self.classifier(embed)

