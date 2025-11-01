"""
train.py
---------
Handles model training, validation, and testing for the Siamese network.
Includes accuracy evaluation, metric tracking, and result plotting.
"""
import torch
import numpy as np
from modules import SiameseNetwork, CombinedLoss, initialize_weights
from dataset import load_data_splits, create_dataloaders
from tqdm import tqdm
from utils import plot_training_curves

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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_benign, train_malignant, test_benign, test_malignant = load_data_splits(
        config['image_dir'],
        config['csv_path'],
        sample_size=config['sample_size'],
        train_ratio=config['train_ratio'],
        seed=config['seed']
    )
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_benign, train_malignant, 
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
        'test_acc': []
    }
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...\n")
    print("=" * 70)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_triplet_loss'].append(train_metrics['triplet_loss'])
        history['train_class_loss'].append(train_metrics['class_loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Total Loss: {train_metrics['loss']:.4f}")
        print(f"  Triplet Loss: {train_metrics['triplet_loss']:.4f}")
        print(f"  Classification Loss: {train_metrics['class_loss']:.4f}")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_metrics['accuracy'],
                'config': config
            }, config['save_path'])
            
            print(f"  ✓ New best model saved! (Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping after {config['patience']} epochs without improvement")
            break
        
        print("=" * 70)
    
    # Plot training curves
    plot_training_curves(history, save_path='training_curves.png')
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Model saved to: {config['save_path']}")
    print(f"{'='*70}\n")


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for anchor, positive, negative, labels in tqdm(test_loader, desc='Evaluating'):
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
