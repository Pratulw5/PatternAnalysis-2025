"""
train.py
---------
Handles model training, validation, and testing for the Siamese network.
Includes accuracy evaluation, metric tracking, and result plotting.
"""
import torch
import numpy as np
import argparse
from modules import SiameseNetwork, CombinedLoss, initialize_weights
from dataset import load_data_splits, create_dataloaders
from tqdm import tqdm
from utils import plot_training_curves, set_seed
 

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Train for one epoch
    
    Returns:
        dict: Training metrics
    """
    model.train()
    epoch_loss = 0.0
    epoch_triplet = 0.0
    epoch_class = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for anchor, positive, negative, labels in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        labels = labels.to(device)
        
        # Forward pass
        anchor_embed, positive_embed, negative_embed = model(
            anchor, positive, negative
        )
        class_logits = model.classifier(anchor_embed)
        
        # Calculate loss
        loss, triplet_loss, class_loss = criterion(
            anchor_embed, positive_embed, negative_embed, 
            labels, class_logits
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate metrics
        epoch_loss += loss.item()
        epoch_triplet += triplet_loss.item()
        epoch_class += class_loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(class_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return {
        'loss': epoch_loss / len(train_loader),
        'triplet_loss': epoch_triplet / len(train_loader),
        'class_loss': epoch_class / len(train_loader),
        'accuracy': correct / total
    }



def train_model(config):
    """
    Main training function
    
    Args:
        config (dict): Training configuration
    """
    # SET SEED FIRST - This is crucial for reproducibility
    set_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    (train_benign, train_malignant, 
     val_benign, val_malignant,
     test_benign, test_malignant) = load_data_splits(
        config['image_dir'],
        config['csv_path'],
        sample_size=config['sample_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_benign, train_malignant, 
        val_benign, val_malignant,
        test_benign, test_malignant,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )
    
    # Create model
    print("\nInitializing model...")
    model = SiameseNetwork(
        embedding_dim=config['embedding_dim'],
        pretrained=True
    ).to(device)
    
    # Initialize new layers
    model = initialize_weights(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        margin=config['margin'],
        alpha=config['alpha']
    )
    
    optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr_backbone']},
        {'params': model.embedding.parameters(), 'lr': config['lr']},
        {'params': model.classifier.parameters(), 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_triplet_loss': [],
        'train_class_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...\n")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validate 
        val_metrics = evaluate(model, val_loader, device, desc='Validating')
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['train_class_loss'].append(train_metrics['class_loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Total Loss:       {train_metrics['loss']:.4f}")
        print(f"  Triplet Loss:     {train_metrics['triplet_loss']:.4f}")
        print(f"  Class Loss:       {train_metrics['class_loss']:.4f}")
        print(f"  Train Accuracy:   {train_metrics['accuracy']:.4f}")
        print(f"  Val Accuracy:     {val_metrics['accuracy']:.4f}")
        print(f"  Learning Rate:    {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model based on validation accuracy
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config
            }, config['save_path'])
            
            print(f"  ✓ New best model saved! (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping after {config['patience']} epochs without improvement")
            break
        
        print("=" * 80)
    
    # Training complete - NO TEST EVALUATION HERE
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Best Val Accuracy:  {best_val_acc:.4f}")
    print(f"  Model saved to: {config['save_path']}")
    print(f"\n  ⚠️  Test set evaluation: Run predict.py for final unbiased performance")
    print("=" * 80)
    
    # Plot training curves (ONLY train and val)
    plot_training_curves(history, save_path='training_curves.png')
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {config['save_path']}")
    print(f"\n  📊 Next step: Run predict.py to evaluate on test set")
    print(f"{'='*80}\n")
    
    return history


def evaluate(model, data_loader, device, desc='Evaluating'):
    """
    Evaluate model on validation set
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        device: torch device
        desc (str): Description for progress bar
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for anchor, positive, negative, labels in tqdm(data_loader, desc=desc):
            anchor = anchor.to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits = model.classify(anchor)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Siamese Network for Melanoma Classification')
    
    # Data parameters
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to metadata CSV')
    parser.add_argument('--sample_size', type=int, default=584,
                        help='Number of samples per class')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training data ratio (default: 0.7 = 70%%)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation data ratio (default: 0.15 = 15%%)')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Weight for triplet loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for new layers')
    parser.add_argument('--lr_backbone', type=float, default=5e-6,
                        help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    # Other parameters
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default='best_model.pth',
                        help='Path to save best model')
    
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    # Train model
    train_model(config)
