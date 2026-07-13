"""
Configuration file for the NAV-SMFS Deep Learning pipeline.
Contains all configurable parameters and paths.
"""

import os
import torch
from pathlib import Path

class Config:
    # Paths - Project root is one level up from ml folder
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_DIR = BASE_DIR / 'raw_data'  # Your raw dataset location
    DATASET_DIR = BASE_DIR / 'dataset'
    MODELS_DIR = BASE_DIR / 'models'
    PLOTS_DIR = BASE_DIR / 'plots'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Create directories if they don't exist
    for dir_path in [DATASET_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Dataset splitting
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Model parameters
    MODEL_NAME = 'efficientnet-b0'
    NUM_CLASSES = 2
    IMG_SIZE = 224
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # Scheduler
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ImageNet normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # File paths
    BEST_MODEL_PATH = MODELS_DIR / 'best_model.pth'
    LAST_MODEL_PATH = MODELS_DIR / 'last_model.pth'
    TRAINING_HISTORY_PATH = LOGS_DIR / 'training_history.json'
    
    # Random seed for reproducibility
    SEED = 42
    
    # Print configuration on load
    print(f"🔧 Configuration loaded:")
    print(f"   Device: {DEVICE}")
    print(f"   Dataset: {DATASET_DIR}")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Plots: {PLOTS_DIR}")
    print(f"   Logs: {LOGS_DIR}")