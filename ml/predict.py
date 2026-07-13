"""
Prediction module for NAV-SMFS.
Single image prediction with confidence score.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
from pathlib import Path

from config import Config
from model import create_model
from utils import setup_logger

class Predictor:
    def __init__(self, model_path=None):
        """Initialize the predictor."""
        self.logger = setup_logger('predictor')
        self.device = Config.DEVICE
        
        # Load model
        if model_path is None:
            model_path = Config.BEST_MODEL_PATH
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = create_model(pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
        
        self.logger.info("Predictor initialized successfully")
    
    def predict_image(self, image_path):
        """
        Predict whether an image is real or AI-generated.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (prediction_class, confidence, probabilities)
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None, None, None
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs.squeeze())
            prob_fake = probabilities.item()
            prob_real = 1 - prob_fake
        
        # Determine prediction
        if prob_fake > 0.5:
            prediction = "AI Generated"
            confidence = prob_fake * 100
        else:
            prediction = "Real"
            confidence = prob_real * 100
        
        return prediction, confidence, {'real': prob_real, 'fake': prob_fake}
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of predictions
        """
        results = []
        for path in image_paths:
            pred, conf, probs = self.predict_image(path)
            results.append({
                'image': str(path),
                'prediction': pred,
                'confidence': conf,
                'probabilities': probs
            })
        return results

def main():
    """Main function for command-line prediction."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = Predictor()
        
        # Make prediction
        prediction, confidence, probabilities = predictor.predict_image(image_path)
        
        # Print results
        print("\n" + "=" * 50)
        print("🔍 PREDICTION RESULT")
        print("=" * 50)
        print(f"Image: {image_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 50)
        print(f"Real Probability: {probabilities['real']*100:.2f}%")
        print(f"Fake Probability: {probabilities['fake']*100:.2f}%")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()