"""
predict.py
-----------
Demonstrates how to load the trained Siamese model and perform
inference on sample image pairs.
"""
import torch 
import numpy as np
from PIL import Image

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

