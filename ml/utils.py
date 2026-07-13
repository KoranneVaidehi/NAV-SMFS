"""
Utility functions for the NAV-SMFS ML pipeline.
Includes image processing, duplicate detection, and helper functions.
"""

import os
import hashlib
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import json
from datetime import datetime
import random
import torch
from config import Config

# Set up logging
def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_image_corrupted(image_path):
    """Check if an image file is corrupted or unreadable."""
    try:
        # Try to open with PIL
        img = Image.open(image_path)
        img.verify()  # Verify integrity
        
        # Try to load with cv2 as a second check
        img = cv2.imread(str(image_path))
        if img is None:
            return True
        
        # Check if image has valid shape
        if len(img.shape) < 2:
            return True
        
        return False
    except Exception as e:
        return True

def get_image_hash(image_path, hash_type='md5'):
    """Generate a hash for an image file."""
    try:
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception:
        return None

def find_duplicates(image_paths):
    """Find duplicate images based on file hashes."""
    hash_map = {}
    duplicates = []
    
    for path in tqdm(image_paths, desc="Checking for duplicates"):
        file_hash = get_image_hash(path)
        if file_hash:
            if file_hash in hash_map:
                duplicates.append((path, hash_map[file_hash]))
            else:
                hash_map[file_hash] = path
    
    return duplicates

def collect_images_from_directory(root_dir):
    """Collect all image paths from a directory recursively."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []
    
    root_path = Path(root_dir)
    for ext in image_extensions:
        image_paths.extend(root_path.rglob(f'*{ext}'))
        image_paths.extend(root_path.rglob(f'*{ext.upper()}'))
    
    return image_paths

def get_class_from_path(image_path, class_mapping):
    """Determine the class of an image based on its path."""
    path_str = str(image_path).lower()
    for class_name, keywords in class_mapping.items():
        if any(keyword in path_str for keyword in keywords):
            return class_name
    return None

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_current_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')