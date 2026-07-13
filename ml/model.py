"""
Model definition for NAV-SMFS.
Uses EfficientNet-B0 with custom classification head for binary classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Use relative import
from .config import Config
from .utils import setup_logger

class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for real/fake face detection."""
    
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Initialize the EfficientNet-B0 model.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(EfficientNetClassifier, self).__init__()
        
        self.logger = setup_logger('model')
        
        # Load pretrained EfficientNet-B0
        self.logger.info(f"Loading EfficientNet-B0 with pretrained={pretrained}")
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
        self.logger.info(f"Model initialized with {num_classes} output classes")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze the backbone layers."""
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            self.logger.info("Backbone layers frozen")
        else:
            self.logger.info("Backbone layers unfrozen")

def create_model(pretrained=True, freeze_backbone=False):
    """
    Create and initialize the EfficientNet classifier.
    
    Args:
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        
    Returns:
        EfficientNetClassifier: The initialized model
    """
    model = EfficientNetClassifier(num_classes=Config.NUM_CLASSES, pretrained=pretrained)
    
    if freeze_backbone:
        model.freeze_backbone(True)
    
    return model

if __name__ == "__main__":
    # Test the model
    model = create_model(pretrained=False)
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")