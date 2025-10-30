"""
modules.py
-----------
Defines all model components for the Siamese Melanoma Detection system.
Includes the Siamese network architecture, embedding layers, and similarity heads.
"""
import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
     """
    Siamese CNN using EfficientNet-B0 as backbone.

    Takes a pair of images and outputs a similarity score (0-1). Produces embeddings
    for each image and computes their absolute difference for classification.

    Args:
        base_model_name (str): Backbone model name, currently only "efficientnet_b0".
        embedding_dim (int): Dimension of the embedding vector.
        freeze_base (bool): Freeze early layers of the backbone if True.
        fine_tune_from_block (int): Block index from which to unfreeze for fine-tuning.
    """
    def __init__(self, base_model_name="efficientnet_b0", embedding_dim=256, freeze_base=True, fine_tune_from_block=5):
        super(SiameseNetwork, self).__init__()

        # Load EfficientNet-B0 backbone
        if base_model_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported base model name")

        # Keep only convolutional features
        self.feature_extractor = nn.Sequential(*list(base_model.features.children()))

        # freeze early layers, allow fine-tuning from a specific block useful while training
        if freeze_base:
            for idx, param in enumerate(self.feature_extractor):
                param.requires_grad = False  # Freeze everything first
            for idx, param in enumerate(self.feature_extractor[fine_tune_from_block:]):
                param.requires_grad = True   # Unfreeze last blocks

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        diff = torch.abs(output1 - output2)
        out = self.classifier(diff)
        return out
