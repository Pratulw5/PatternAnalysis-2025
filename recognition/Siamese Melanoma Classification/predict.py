"""
predict.py
-----------
Demonstrates how to load the trained Siamese model and perform
inference on sample image pairs.
"""
import torch 
def evaluate_siamese(model, similarity_layer, data_loader, device, threshold=0.5):
    """
    Evaluate the Siamese network with a similarity head on a dataset.

    Args:
        model (nn.Module): Siamese base model.
        similarity_layer (nn.Module): Linear similarity head.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run evaluation on.
        threshold (float): Probability threshold to classify as similar.

    Returns:
        float: Accuracy on the given dataset.
    """
    model.eval()
    similarity_layer.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in data_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            embed1 = model.forward_once(img1)
            embed2 = model.forward_once(img2)
            diff = torch.abs(embed1 - embed2)

            logits = similarity_layer(diff)
            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).float()
            correct += (preds == labels.view(-1,1)).sum().item()
            total += labels.size(0)

    model.train()
    similarity_layer.train()
    return correct / total
