"""
Model loader for NAV-SMFS.
Loads the model once at startup and provides prediction functions.
This ensures the model is not reloaded for every request.
"""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Use relative imports
from .config import Config
from .model import create_model
from .utils import setup_logger

class ModelLoader:
    """Singleton class to load and serve the model."""
    
    _instance = None
    _model = None
    _device = None
    _transform = None
    _idx_to_class = None
    _class_to_idx = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance."""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model, device, and transforms."""
        self.logger = setup_logger('model_loader')
        self.logger.info("=" * 60)
        self.logger.info("Initializing Model Loader...")
        self.logger.info("=" * 60)
        
        # Set device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device: {self._device}")
        
        # Load model
        model_path = Config.BEST_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self._device)
        
        self._model = create_model(pretrained=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = self._model.to(self._device)
        self._model.eval()
        
        # Get class mapping from dataset
        self._class_to_idx = self._get_class_mapping()
        self._idx_to_class = {v: k for k, v in self._class_to_idx.items()}
        
        self.logger.info(f"Class mapping: {self._class_to_idx}")
        self.logger.info(f"Index to class: {self._idx_to_class}")
        
        # Define preprocessing transforms
        self._transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
        
        self.logger.info("Model Loader initialized successfully!")
        self.logger.info("=" * 60)
    
    def _get_class_mapping(self):
        """Get class mapping from dataset directory."""
        dataset_dir = Config.DATASET_DIR / 'train'
        
        if not dataset_dir.exists():
            self.logger.warning(f"Dataset directory not found: {dataset_dir}")
            # Fallback to alphabetical order
            return {'fake': 0, 'real': 1}
        
        # Get class folders and sort alphabetically
        class_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]
        class_folders.sort()
        
        class_to_idx = {}
        for idx, folder in enumerate(class_folders):
            class_name = folder.name.lower()
            class_to_idx[class_name] = idx
        
        return class_to_idx
    
    def predict_image(self, image_path):
        """
        Predict whether an image is real or AI-generated.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: {
                "prediction": "Real" or "AI Generated",
                "confidence": float,
                "real_probability": float,
                "fake_probability": float
            }
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "real_probability": 0.0,
                "fake_probability": 0.0,
                "error": str(e)
            }
        
        # Preprocess
        image_tensor = self._transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self._device)
        
        # Predict
        with torch.no_grad():
            outputs = self._model(image_tensor)
            
            # Get probabilities using softmax
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction (class with highest probability)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            # Convert to Python values
            prediction_idx = prediction.item()
            confidence = confidence.item() * 100
            
            # Get class name from index
            class_name = self._idx_to_class[prediction_idx]
            prediction_text = class_name.capitalize()
            
            # Map to user-friendly labels
            if prediction_text.lower() == 'fake':
                prediction_text = 'AI Generated'
            
            # Get individual class probabilities
            real_prob = probabilities[0, self._class_to_idx['real']].item() * 100
            fake_prob = probabilities[0, self._class_to_idx['fake']].item() * 100
        
        return {
            "prediction": prediction_text,
            "confidence": confidence,
            "real_probability": real_prob,
            "fake_probability": fake_prob
        }
    
    def get_model_info(self):
        """Get model information."""
        return {
            "device": str(self._device),
            "class_mapping": self._class_to_idx,
            "num_classes": len(self._class_to_idx)
        }

# Global instance for easy import
model_loader = ModelLoader()

def predict_image(image_path):
    """
    Convenience function to predict a single image.
    Uses the globally loaded model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Prediction results
    """
    return model_loader.predict_image(image_path)