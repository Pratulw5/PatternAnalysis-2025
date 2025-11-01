import matplotlib.pyplot as plt
from modules import SiameseNetwork
import torch

def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training curves
    
    Args:
        history (dict): Training history
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Triplet loss
    axes[0, 1].plot(history['train_triplet_loss'], label='Triplet Loss', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Triplet Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Classification loss
    axes[1, 0].plot(history['train_class_loss'], label='Classification Loss', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Accuracy
    axes[1, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[1, 1].plot(history['test_acc'], label='Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
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
    print(f"Test accuracy: {checkpoint.get('test_acc', 'N/A'):.4f}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model