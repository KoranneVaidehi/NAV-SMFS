"""
NAV-SMFS Machine Learning Pipeline
Deepfake Face Detection using EfficientNet-B0
"""

from .config import Config
from .dataset import FaceDataset, create_dataloaders, get_transforms
from .model import EfficientNetClassifier, create_model
from .utils import setup_logger, set_seed
from .model_loader import ModelLoader, predict_image

__version__ = "1.0.0"

# Print initialization info
print(f"ML Module v{__version__} loaded successfully!")