"""
Dataset and DataLoader modules for NAV-SMFS.
Handles image loading, preprocessing, and augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from config import Config
from utils import setup_logger

class FaceDataset(Dataset):
    """Face dataset for real/fake classification."""
    
    def __init__(self, data_dir, transform=None, is_train=False):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the dataset directory (train/val/test)
            transform: Optional transform to apply to images
            is_train: Whether this is training dataset (affects augmentation)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.logger = setup_logger(f'dataset_{data_dir.name}')
        
        # Collect all images and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Get class folders
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        class_folders = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_folders.sort()  # Ensure consistent ordering
        
        if not class_folders:
            raise ValueError(f"No class folders found in {self.data_dir}")
        
        # Create class mappings
        for idx, folder in enumerate(class_folders):
            class_name = folder.name.lower()
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Collect all images in this class
            for img_path in folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    self.samples.append((img_path, idx))
        
        # Log dataset statistics
        self.logger.info(f"Loaded {len(self.samples)} images from {data_dir}")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == idx)
            self.logger.info(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            self.logger.warning(f"Error loading image {img_path}: {e}")
            # Return a placeholder image (black image) to avoid breaking the batch
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_train=False):
    """
    Get image transforms based on whether it's training or not.
    
    Args:
        is_train: Whether to include training augmentations
        
    Returns:
        transforms.Compose: The transform pipeline
    """
    if is_train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
    
    return transform

def create_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=4):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset_dir = Config.DATASET_DIR
    
    # Create datasets
    train_dataset = FaceDataset(
        dataset_dir / 'train',
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    val_dataset = FaceDataset(
        dataset_dir / 'val',
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    test_dataset = FaceDataset(
        dataset_dir / 'test',
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset and dataloaders
    print("Testing dataset and dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break