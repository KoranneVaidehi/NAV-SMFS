"""
Model loader for NAV-SMFS.
Loads the model once at startup and provides prediction functions.
"""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from .config import Config
from .model import create_model
from .utils import setup_logger
from .gradcam import generate_gradcam_for_face


class ModelLoader:
    _instance = None
    _model = None
    _device = None
    _transform = None
    _idx_to_class = None
    _class_to_idx = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.logger = setup_logger('model_loader')
        self.logger.info("=" * 60)
        self.logger.info("Initializing Model Loader...")
        self.logger.info("=" * 60)
        
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device: {self._device}")
        
        model_path = Config.BEST_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self._device)
        
        self._model = create_model(pretrained=False)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = self._model.to(self._device)
        self._model.eval()
        
        self._class_to_idx = self._get_class_mapping()
        self._idx_to_class = {v: k for k, v in self._class_to_idx.items()}
        
        self.logger.info(f"Class mapping: {self._class_to_idx}")
        self.logger.info(f"Index to class: {self._idx_to_class}")
        
        self._transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
        
        self.logger.info("Model Loader initialized successfully!")
        self.logger.info("=" * 60)
    
    def _get_class_mapping(self):
        dataset_dir = Config.DATASET_DIR / 'train'
        if not dataset_dir.exists():
            return {'fake': 0, 'real': 1}
        
        class_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]
        class_folders.sort()
        
        class_to_idx = {}
        for idx, folder in enumerate(class_folders):
            class_name = folder.name.lower()
            class_to_idx[class_name] = idx
        
        return class_to_idx
    
    def predict_image(self, image_path):
        if self._model is None:
            raise RuntimeError("Model not initialized.")
        
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
        
        image_tensor = self._transform(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            outputs = self._model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            prediction_idx = prediction.item()
            confidence = confidence.item() * 100
            
            class_name = self._idx_to_class[prediction_idx]
            prediction_text = class_name.capitalize()
            
            if prediction_text.lower() == 'fake':
                prediction_text = 'AI Generated'
            
            real_prob = probabilities[0, self._class_to_idx['real']].item() * 100
            fake_prob = probabilities[0, self._class_to_idx['fake']].item() * 100
        
        return {
            "prediction": prediction_text,
            "confidence": confidence,
            "real_probability": real_prob,
            "fake_probability": fake_prob
        }
    
    def generate_gradcam(self, face_image_path, target_class=None):
        """Generate Grad-CAM for a face image."""
        if self._model is None:
            raise RuntimeError("Model not initialized.")
        
        print("\n🔥 ModelLoader.generate_gradcam()")
        print(f"   Face path: {face_image_path}")
        print(f"   Target class: {target_class}")
        
        result = generate_gradcam_for_face(
            self._model,
            face_image_path,
            self._device,
            target_class
        )
        
        return result
    
    def generate_heatmap_overlay(self, face_image_path, output_path=None, target_class=None):
        """Generate and save heatmap overlay for a face image."""
        print("\n🔥 ModelLoader.generate_heatmap_overlay()")
        print(f"   Face path: {face_image_path}")
        print(f"   Output path: {output_path}")
        print(f"   Target class: {target_class}")
        
        result = self.generate_gradcam(face_image_path, target_class)
        
        if result.get('success', False) and result.get('overlay') is not None and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_bgr = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), overlay_bgr)
            result['saved_path'] = str(output_path)
            print(f"   ✅ Overlay saved: {output_path}")
        elif output_path is None:
            print(f"   ⚠️ No output_path provided")
        
        return result


model_loader = ModelLoader()

def predict_image(image_path):
    return model_loader.predict_image(image_path)