"""
Dataset preparation module for NAV-SMFS.
Automatically analyzes raw dataset, handles corrupted images,
removes duplicates, and creates train/val/test splits.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from utils import (
    setup_logger, collect_images_from_directory, is_image_corrupted,
    find_duplicates, get_class_from_path, save_json, set_seed
)

class DatasetPreparer:
    def __init__(self):
        self.logger = setup_logger('dataset_preparer', Config.LOGS_DIR / 'dataset_preparation.log')
        self.raw_dir = Config.RAW_DATA_DIR
        self.dataset_dir = Config.DATASET_DIR
        self.class_mapping = {
            'real': ['real', 'genuine', 'authentic', 'original'],
            'fake': ['fake', 'synthetic', 'generated', 'ai', 'gan', 'deepfake']
        }
        set_seed(Config.SEED)
        
    def analyze_dataset(self):
        """Analyze the raw dataset structure and identify classes."""
        self.logger.info("=" * 60)
        self.logger.info(f"Analyzing dataset at: {self.raw_dir}")
        self.logger.info("=" * 60)
        
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        
        # Collect all images
        all_images = collect_images_from_directory(self.raw_dir)
        self.logger.info(f"Found {len(all_images)} total images")
        
        # Identify classes from folder structure
        classes = set()
        for img_path in all_images:
            class_name = get_class_from_path(img_path, self.class_mapping)
            if class_name:
                classes.add(class_name)
        
        if not classes:
            # If no classes found, try to infer from parent folder names
            parent_folders = [img.parent.name.lower() for img in all_images]
            for folder in set(parent_folders):
                if any(keyword in folder for keyword in ['real', 'genuine', 'authentic', 'original']):
                    classes.add('real')
                elif any(keyword in folder for keyword in ['fake', 'synthetic', 'generated', 'ai', 'gan']):
                    classes.add('fake')
        
        self.classes = list(classes)
        self.logger.info(f"Identified classes: {self.classes}")
        
        if len(self.classes) != 2:
            self.logger.warning(f"Expected 2 classes (real/fake), found: {self.classes}")
            self.logger.warning("Attempting to auto-detect classes...")
            # Auto-detect using folder names containing 'real' or 'fake'
            auto_classes = set()
            for img_path in all_images:
                path_str = str(img_path).lower()
                if 'real' in path_str or 'genuine' in path_str:
                    auto_classes.add('real')
                elif 'fake' in path_str or 'synthetic' in path_str or 'ai' in path_str:
                    auto_classes.add('fake')
            self.classes = list(auto_classes) if len(auto_classes) == 2 else ['real', 'fake']
            self.logger.info(f"Auto-detected classes: {self.classes}")
        
        # Count images per class
        class_counts = {cls: 0 for cls in self.classes}
        for img_path in all_images:
            class_name = get_class_from_path(img_path, self.class_mapping)
            if class_name in class_counts:
                class_counts[class_name] += 1
        
        for cls, count in class_counts.items():
            self.logger.info(f"  {cls}: {count} images")
        
        return all_images
    
    def clean_dataset(self, image_paths):
        """Remove corrupted images and duplicates."""
        self.logger.info("=" * 60)
        self.logger.info("Cleaning dataset...")
        self.logger.info("=" * 60)
        
        # Remove corrupted images
        valid_images = []
        corrupted_count = 0
        
        for img_path in tqdm(image_paths, desc="Checking for corrupted images"):
            if not is_image_corrupted(img_path):
                valid_images.append(img_path)
            else:
                corrupted_count += 1
                self.logger.debug(f"Corrupted image: {img_path}")
        
        self.logger.info(f"Removed {corrupted_count} corrupted images")
        
        # Find and remove duplicates
        self.logger.info("Checking for duplicate images...")
        duplicates = find_duplicates(valid_images)
        duplicate_count = len(duplicates)
        
        # Remove duplicates (keep the first occurrence)
        duplicate_paths = {dup[0] for dup in duplicates}
        cleaned_images = [img for img in valid_images if img not in duplicate_paths]
        
        self.logger.info(f"Removed {duplicate_count} duplicate images")
        self.logger.info(f"Total valid images after cleaning: {len(cleaned_images)}")
        
        if len(cleaned_images) < 10:
            raise ValueError(f"Too few images after cleaning: {len(cleaned_images)}")
        
        return cleaned_images
    
    def create_directory_structure(self):
        """Create the train/val/test directory structure."""
        self.logger.info("Creating directory structure...")
        splits = ['train', 'val', 'test']
        
        for split in splits:
            for class_name in self.classes:
                split_dir = self.dataset_dir / split / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {split_dir}")
    
    def split_dataset(self, image_paths):
        """Split dataset into train/val/test sets."""
        self.logger.info("=" * 60)
        self.logger.info("Splitting dataset...")
        self.logger.info("=" * 60)
        
        # Group images by class
        class_images = {class_name: [] for class_name in self.classes}
        
        for img_path in image_paths:
            class_name = get_class_from_path(img_path, self.class_mapping)
            if class_name in class_images:
                class_images[class_name].append(img_path)
            else:
                # Try to assign based on class name in path
                for cls in self.classes:
                    if cls in str(img_path).lower():
                        class_images[cls].append(img_path)
                        break
        
        # Log class distribution
        for class_name, images in class_images.items():
            self.logger.info(f"Class '{class_name}': {len(images)} images")
        
        # Split each class
        splits = {}
        for class_name, images in class_images.items():
            if not images:
                self.logger.warning(f"No images found for class '{class_name}'")
                continue
            
            # First split: train vs temp (val+test)
            train, temp = train_test_split(
                images,
                test_size=(1 - Config.TRAIN_RATIO),
                random_state=Config.SEED,
                shuffle=True
            )
            
            # Split temp into val and test
            val_ratio = Config.VAL_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
            val, test = train_test_split(
                temp,
                test_size=(1 - val_ratio),
                random_state=Config.SEED,
                shuffle=True
            )
            
            splits[class_name] = {
                'train': train,
                'val': val,
                'test': test
            }
            
            self.logger.info(
                f"Class '{class_name}' - Train: {len(train)}, "
                f"Val: {len(val)}, Test: {len(test)}"
            )
        
        return splits
    
    def copy_images_to_dataset(self, splits):
        """Copy images to the dataset directory."""
        self.logger.info("=" * 60)
        self.logger.info("Copying images to dataset directory...")
        self.logger.info("=" * 60)
        
        total_copied = 0
        
        for class_name, class_splits in splits.items():
            for split_name, images in class_splits.items():
                dest_dir = self.dataset_dir / split_name / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in tqdm(images, desc=f"Copying {split_name}/{class_name}"):
                    dest_path = dest_dir / img_path.name
                    # Handle duplicate filenames
                    counter = 1
                    while dest_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    shutil.copy2(img_path, dest_path)
                    total_copied += 1
        
        self.logger.info(f"Total images copied: {total_copied}")
        
        # Save split statistics
        stats = {
            'total_images': total_copied,
            'classes': self.classes,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            stats['splits'][split] = {}
            for class_name in self.classes:
                count = len(list((self.dataset_dir / split / class_name).glob('*')))
                stats['splits'][split][class_name] = count
        
        stats_file = self.dataset_dir / 'dataset_stats.json'
        save_json(stats, stats_file)
        self.logger.info(f"Dataset statistics saved to: {stats_file}")
        
        # Print summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Dataset Summary:")
        for split in ['train', 'val', 'test']:
            total = sum(stats['splits'][split].values())
            self.logger.info(f"  {split}: {total} images")
            for class_name in self.classes:
                count = stats['splits'][split][class_name]
                self.logger.info(f"    {class_name}: {count}")
        self.logger.info("=" * 60)
    
    def prepare_dataset(self):
        """Main method to prepare the complete dataset."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🚀 STARTING DATASET PREPARATION FOR NAV-SMFS")
        self.logger.info("=" * 60 + "\n")
        
        # Analyze dataset
        all_images = self.analyze_dataset()
        
        # Clean dataset
        cleaned_images = self.clean_dataset(all_images)
        
        # Create directory structure
        self.create_directory_structure()
        
        # Split dataset
        splits = self.split_dataset(cleaned_images)
        
        # Copy images to dataset directory
        self.copy_images_to_dataset(splits)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ DATASET PREPARATION COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 60 + "\n")
        
        return splits

def main():
    """Main function to run dataset preparation."""
    preparer = DatasetPreparer()
    preparer.prepare_dataset()

if __name__ == "__main__":
    main()