"""
predict.py
-----------
Demonstrates how to load the trained Siamese model and perform
inference on sample image pairs.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from dataset import get_transforms, load_data_splits, TripletMelanomaDataset
from utils import load_model

def predict_single_image(model, image_path, transform, device):
    """
    Predict class for a single image
    
    Args:
        model: Trained model
        image_path (str): Path to image
        transform: Image transforms
        device: torch device
        
    Returns:
        tuple: (predicted_class, probability)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model.classify(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence


def evaluate_and_visualize(model, test_loader, device, save_dir='results'):
    """
    Evaluate model and create visualizations
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        save_dir (str): Directory to save results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for anchor, _, _, labels in test_loader:
            anchor = anchor.to(device)
            
            logits = model.classify(anchor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds, 
        target_names=['Benign', 'Malignant'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_dir}/confusion_matrix.png")
    plt.close()
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    
    benign_probs = all_probs[all_labels == 0, 0]
    malignant_probs = all_probs[all_labels == 1, 1]
    
    plt.hist(benign_probs, bins=30, alpha=0.5, label='Benign', color='blue')
    plt.hist(malignant_probs, bins=30, alpha=0.5, label='Malignant', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Confidence distribution saved to {save_dir}/confidence_distribution.png")
    plt.close()
    
    print("="*70 + "\n")


def prediction(checkpoint_path, image_dir, csv_path):
    """
    Run demo inference on test images
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        image_dir (str): Directory containing images
        csv_path (str): Path to metadata CSV
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_benign, test_malignant = load_data_splits(
        image_dir, csv_path, sample_size=584, train_ratio=0.8, seed=42
    )
    
    # Combine test paths and labels
    test_paths = test_benign + test_malignant
    test_labels = [0] * len(test_benign) + [1] * len(test_malignant)
    
    # Get transforms
    transform = get_transforms(img_size=224, mode='test')
    
    # Create test loader for full evaluation
    test_dataset = TripletMelanomaDataset(
        test_benign, test_malignant,
        transform=transform,
        num_triplets=4000,
        seed=123
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    
    # Full evaluation
    evaluate_and_visualize(model, test_loader, device, save_dir='results')
    
    # Single image example
    print("\nSingle Image Prediction Example:")
    print("-" * 70)
    example_path = test_paths[0]
    pred_class, confidence = predict_single_image(model, example_path, transform, device)
    class_names = ['Benign', 'Malignant']
    print(f"Image: {example_path}")
    print(f"Predicted: {class_names[pred_class]}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained Siamese Network')
    
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to metadata CSV')
    
    args = parser.parse_args()
    
    prediction(args.checkpoint, args.image_dir, args.csv_path)
