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
            nn.Linear(1280, 512),
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
        return F.normalize(x, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet learning
        
        Args:
            anchor (torch.Tensor): Anchor images
            positive (torch.Tensor): Positive examples (same class as anchor)
            negative (torch.Tensor): Negative examples (different class)
            
        Returns:
            tuple: (anchor_embed, positive_embed, negative_embed)
        """
        anchor_embed = self.forward_once(anchor)
        positive_embed = self.forward_once(positive)
        negative_embed = self.forward_once(negative)
        return anchor_embed, positive_embed, negative_embed

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

def initialize_weights(model, seed=42):
    """
    Initialize weights of the embedding and classifier layers using Kaiming Normal initialization.
    
    Args:
        model: PyTorch model
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    def init_func(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Only initialize embedding and classifier (not pretrained backbone)
    model.embedding.apply(init_func)
    model.classifier.apply(init_func)
    
    return model


class CombinedLoss(nn.Module):
    """
    Combined loss function using both triplet loss and classification loss
    """
    
    def __init__(self, margin=1.0, alpha=0.6):
        """
        Args:
            margin (float): Margin for triplet loss
            alpha (float): Weight for triplet loss (1-alpha for classification)
        """
        super(CombinedLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, anchor_embed, positive_embed, negative_embed, 
                labels, class_logits):
        """
        Calculate combined loss
        
        Args:
            anchor_embed: Anchor embeddings
            positive_embed: Positive embeddings
            negative_embed: Negative embeddings
            labels: Ground truth labels
            class_logits: Classification predictions
            
        Returns:
            tuple: (total_loss, triplet_loss, classification_loss)
        """
        # Triplet loss: anchor closer to positive than negative
        triplet = self.triplet_loss(anchor_embed, positive_embed, negative_embed)
        
        # Classification loss
        classification = self.ce_loss(class_logits, labels)
        
        # Combined weighted loss
        total_loss = self.alpha * triplet + (1 - self.alpha) * classification
        
        return total_loss, triplet, classification