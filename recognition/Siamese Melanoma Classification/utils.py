import matplotlib.pyplot as plt
from modules import SiameseNetwork
import torch
import numpy as np
import os
import random

def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training curves including validation metrics
    
    Args:
        history (dict): Training history
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Triplet loss
    axes[0, 1].plot(history['train_triplet_loss'], label='Triplet Loss', 
                    color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Triplet Loss', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Classification loss
    axes[1, 0].plot(history['train_class_loss'], label='Classification Loss', 
                    color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].set_title('Classification Loss', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy - Train and Val only (NO TEST)
    axes[1, 1].plot(history['train_acc'], label='Train Accuracy', 
                    linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(history['val_acc'], label='Val Accuracy', 
                    linewidth=2, marker='s', markersize=4)
    
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy', fontsize=11)
    axes[1, 1].set_title('Model Accuracy (Train & Validation)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device: torch device
        
    Returns:
        model: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    embedding_dim = config.get('embedding_dim', 256)
    
    # Initialize model
    model = SiameseNetwork(
        embedding_dim=embedding_dim,
        pretrained=False  # We're loading weights
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model
def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)                    # Python random
    np.random.seed(seed)                 # Numpy
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed(seed)         # PyTorch single GPU
    torch.cuda.manual_seed_all(seed)     # PyTorch multi-GPU
    
    # Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def worker_init_fn(worker_id):
    """Initialize worker with unique but reproducible seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)