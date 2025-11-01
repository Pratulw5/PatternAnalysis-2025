"""
train.py
---------
Handles model training, validation, and testing for the Siamese network.
Includes accuracy evaluation, metric tracking, and result plotting.
"""
import torch
import numpy as np
from modules import SiameseNetwork
from dataset import SiameseMelanomaDataset
from modules import SiameseNetwork, initialize_weights
from dataset import load_data_splits, create_dataloaders
from tqdm import tqdm

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


class SiameseTrainer:
    """
    Trainer class for SiameseNetwork with contrastive loss.

    Handles dataset creation, dataloaders, model initialization, and training.
    """
    def __init__(self, train_benign, train_malignant, test_benign, test_malignant,
                 transform=None, embedding_dim=256, freeze_base=True,
                 fine_tune_from_block=2, batch_size=16, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.rng = np.random.default_rng(42)

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
            if "embedding" in name:
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

    def forward(self, x1, x2):
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        emb1, emb2 = self.model(x1, x2)
        return emb1, emb2

    def train(self, epochs=5, lr=1e-4, margin=1.0):
        """Main training loop for contrastive loss."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for img1, img2, label in self.train_loader:
                emb1, emb2 = self.forward(img1, img2)
                label = label.to(self.device)
                loss = self.model.contrastive_loss(emb1, emb2, label, margin)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")



if __name__ == "__main__":

    train_benign, train_malignant = ["path/to/benign1.jpg"], ["path/to/malignant1.jpg"]
    test_benign, test_malignant = ["path/to/benign_test.jpg"], ["path/to/malignant_test.jpg"]
    IMG_SIZE = 224

    trainer = SiameseTrainer(train_benign, train_malignant,test_benign, test_malignanttransform=SiameseMelanomaDataset.transform, batch_size=4)

    trainer.train(epochs=2, lr=1e-4)
