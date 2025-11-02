"""
predict.py
-----------
The test set is ONLY used here, AFTER all training and tuning is complete.
This provides the final, unbiased performance estimate.
Author: Pratul Wadhwa
Student_Id: 49073085
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
    Evaluate model on TEST SET and create comprehensive visualizations
    
     This is the ONLY place where test set is evaluated!
    
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
    
    print("\n" + "=" * 70)
    print(" EVALUATING ON TEST SET (FINAL UNBIASED PERFORMANCE)")
    print("=" * 70)
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
    print(" TEST SET RESULTS (FINAL PERFORMANCE)")
    print("="*70)
    print(f"\n Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n Classification Report:")
    print(classification_report(
        all_labels, all_preds, 
        target_names=['Benign', 'Malignant'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 6))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # 2. Confidence Distribution
    ax2 = plt.subplot(1, 3, 2)
    benign_probs = all_probs[all_labels == 0, 0]
    malignant_probs = all_probs[all_labels == 1, 1]
    
    ax2.hist(benign_probs, bins=30, alpha=0.6, label='Benign', color='blue', edgecolor='black')
    ax2.hist(malignant_probs, bins=30, alpha=0.6, label='Malignant', color='red', edgecolor='black')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-Class Accuracy
    ax3 = plt.subplot(1, 3, 3)
    benign_correct = (all_preds[all_labels == 0] == 0).sum()
    benign_total = (all_labels == 0).sum()
    malignant_correct = (all_preds[all_labels == 1] == 1).sum()
    malignant_total = (all_labels == 1).sum()
    
    benign_acc = benign_correct / benign_total
    malignant_acc = malignant_correct / malignant_total
    
    bars = ax3.bar(['Benign', 'Malignant'], [benign_acc, malignant_acc], 
                   color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, [benign_acc, malignant_acc]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}\n({acc*100:.2f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_results_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Comprehensive test results saved to {save_dir}/test_results_comprehensive.png")
    plt.close()
    
    print("="*70 + "\n")
    
    return accuracy, cm


def visualize_predictions(model, image_paths, labels, transform, device, 
                         save_path='sample_predictions.png', num_samples=8):
    """
    Visualize predictions on sample test images
    
    Args:
        model: Trained model
        image_paths (list): List of image paths
        labels (list): True labels
        transform: Image transforms
        device: torch device
        save_path (str): Path to save visualization
        num_samples (int): Number of samples to visualize
    """
    # Select random samples
    np.random.seed(42)  # For reproducible visualization
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    class_names = ['Benign', 'Malignant']
    
    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off')
            continue
        
        img_idx = indices[idx]
        img_path = image_paths[img_idx]
        true_label = labels[img_idx]
        
        # Load image for display
        image = Image.open(img_path).convert('RGB')
        
        # Get prediction
        pred_class, confidence = predict_single_image(
            model, img_path, transform, device
        )
        
        # Display
        ax.imshow(image)
        ax.axis('off')
        
        # Color code: green if correct, red if wrong
        color = 'green' if pred_class == true_label else 'red'
        title = f'True: {class_names[true_label]}\n'
        title += f'Pred: {class_names[pred_class]} ({confidence:.2%})'
        ax.set_title(title, color=color, fontweight='bold', fontsize=11)
    
    plt.suptitle('Sample Test Set Predictions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Sample predictions saved to {save_path}")
    plt.close()



def prediction(checkpoint_path, image_dir, csv_path, train_ratio=0.7, val_ratio=0.15):
    """
     FINAL TEST SET EVALUATION
    
    This function evaluates the trained model on the test set.
    The test set has NEVER been seen during training or validation.
    This provides an unbiased estimate of model performance.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        image_dir (str): Directory containing images
        csv_path (str): Path to metadata CSV
        train_ratio (float): Training data ratio
        val_ratio (float): Validation data ratio
    """
    print("\n" + "=" * 70)
    print(" FINAL TEST SET EVALUATION - UNBIASED PERFORMANCE")
    print("=" * 70)
    print("  The test set has NEVER been used during training/tuning")
    print("=" * 70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading trained model...")
    model = load_model(checkpoint_path, device)
    
    # Load test data
    print("\nLoading test data...")
    (_, _, _, _, test_benign, test_malignant) = load_data_splits(
        image_dir, csv_path, 
        sample_size=584, 
        train_ratio=train_ratio, 
        val_ratio=val_ratio, 
        seed=42
    )
    
    print(f"Test set size: {len(test_benign) + len(test_malignant)} images")
    print(f"  - Benign: {len(test_benign)}")
    print(f"  - Malignant: {len(test_malignant)}")
    
    # Combine test paths and labels
    test_paths = test_benign + test_malignant
    test_labels = [0] * len(test_benign) + [1] * len(test_malignant)
    
    # Get transforms
    transform = get_transforms(img_size=224, mode='test')
    
    # Create test loader
    test_dataset = TripletMelanomaDataset(
        test_benign, test_malignant,
        transform=transform,
        num_triplets=4000,
        seed=456
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    
    # Full test evaluation with comprehensive visualizations
    test_accuracy, confusion_mat = evaluate_and_visualize(
        model, test_loader, device, save_dir='results'
    )
    
    # Single image example
    print("\n" + "=" * 70)
    print(" Single Image Prediction Example")
    print("=" * 70)
    example_path = test_paths[0]
    pred_class, confidence = predict_single_image(model, example_path, transform, device)
    class_names = ['Benign', 'Malignant']
    print(f"Image: {example_path}")
    print(f"Predicted: {class_names[pred_class]}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("=" * 70)
    
    # Final summary
    print("\n" + "=" * 70)
    print(" TEST SET EVALUATION COMPLETE")
    print("=" * 70)
    print(f" Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f" All results saved to 'results/' directory")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='🧪 Evaluate trained Siamese Network on TEST SET (Final Unbiased Performance)'
    )
    
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to metadata CSV')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training data ratio (must match training)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation data ratio (must match training)')
    
    args = parser.parse_args()
    
    prediction(args.checkpoint, args.image_dir, args.csv_path, 
               args.train_ratio, args.val_ratio)
