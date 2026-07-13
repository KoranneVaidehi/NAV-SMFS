"""
Training pipeline for NAV-SMFS.
Handles model training, validation, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime

from config import Config
from model import create_model
from dataset import create_dataloaders
from utils import setup_logger, set_seed, save_json

class Trainer:
    def __init__(self):
        """Initialize the trainer with configuration."""
        self.logger = setup_logger('trainer', Config.LOGS_DIR / 'training.log')
        self.device = Config.DEVICE
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        set_seed(Config.SEED)
        
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 TRAINING STARTED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch Size: {Config.BATCH_SIZE}")
        self.logger.info(f"Epochs: {Config.EPOCHS}")
        self.logger.info(f"Learning Rate: {Config.LEARNING_RATE}")
        self.logger.info("=" * 60)
    
    def create_dataloaders(self):
        """Create train, validation, and test dataloaders."""
        self.logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders()
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")
    
    def create_model(self):
        """Create and initialize the model."""
        self.logger.info("Creating model...")
        self.model = create_model(pretrained=True, freeze_backbone=False)
        self.model = self.model.to(self.device)
        self.logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        self.logger.info("Setting up optimizer and scheduler...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.logger.info("Optimizer and scheduler initialized")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.long().to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.long().to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': {
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'weight_decay': Config.WEIGHT_DECAY,
            }
        }
        
        # Save last checkpoint
        torch.save(checkpoint, Config.LAST_MODEL_PATH)
        self.logger.info(f"Checkpoint saved to {Config.LAST_MODEL_PATH}")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, Config.BEST_MODEL_PATH)
            self.logger.info(f"Best model saved to {Config.BEST_MODEL_PATH}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint for resuming training."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Resuming from epoch {start_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return start_epoch
    
    def train(self, resume_from=None):
        """Main training loop."""
        # Initialize everything
        self.create_dataloaders()
        self.create_model()
        self.setup_optimizer_and_scheduler()
        
        start_epoch = 0
        
        # Resume training if checkpoint provided
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(resume_from)
        
        # Training loop
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Starting training...")
        self.logger.info("=" * 60 + "\n")
        
        for epoch in range(start_epoch, Config.EPOCHS):
            self.logger.info(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            self.logger.info("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Save training history
            save_json(self.history, Config.TRAINING_HISTORY_PATH)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check for improvement
            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                is_best = True
                self.logger.info(f"🎉 New best validation accuracy: {val_acc:.2f}%")
            else:
                self.early_stopping_counter += 1
                self.logger.info(f"Early stopping counter: {self.early_stopping_counter}/{Config.EARLY_STOPPING_PATIENCE}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ TRAINING COMPLETED!")
        self.logger.info(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 60)
        
        return self.history

def main():
    """Main function to run training."""
    trainer = Trainer()
    
    # Check if resume is requested
    resume_from = None
    if Config.LAST_MODEL_PATH.exists():
        response = input("Found existing checkpoint. Resume training? (y/n): ")
        if response.lower() == 'y':
            resume_from = Config.LAST_MODEL_PATH
    
    trainer.train(resume_from=resume_from)

if __name__ == "__main__":
    main()