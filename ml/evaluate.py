"""
Evaluation module for NAV-SMFS.
Computes comprehensive metrics and generates plots.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

from config import Config
from model import create_model
from dataset import create_dataloaders
from utils import setup_logger, save_json

class Evaluator:
    def __init__(self):
        """Initialize the evaluator."""
        self.logger = setup_logger('evaluator', Config.LOGS_DIR / 'evaluation.log')
        self.device = Config.DEVICE
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION STARTED")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)
        
        # Storage for predictions and labels
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def load_model(self, model_path=None):
        """Load the trained model."""
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
        
        self.logger.info("Model loaded successfully")
        
        # Load training history if available
        if Config.TRAINING_HISTORY_PATH.exists():
            with open(Config.TRAINING_HISTORY_PATH, 'r') as f:
                self.history = json.load(f)
    
    def evaluate(self, data_loader, dataset_name="Test"):
        """Evaluate model on a dataset."""
        self.logger.info(f"\nEvaluating on {dataset_name} dataset...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                images = images.to(self.device)
                labels = labels.cpu().numpy()
                
                # Forward pass
                outputs = self.model(images)
                
                # Get probabilities using softmax
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions (class with highest probability)
                preds = torch.argmax(outputs, dim=1)
                
                # Store predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels)
                
                # Store ONLY the probability of the Fake class (class 1)
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Store for later use
        if dataset_name.lower() == "test":
            self.all_preds = all_preds
            self.all_labels = all_labels
            self.all_probs = all_probs
        
        # Compute metrics
        metrics = self.compute_metrics(all_labels, all_preds, all_probs)
        
        self.logger.info(f"\n{dataset_name} Metrics:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics, all_labels, all_preds, all_probs
    
    def compute_metrics(self, y_true, y_pred, y_prob):
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Probability of class 1 (Fake) (N,)
        
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.5
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix."""
        if save_path is None:
            save_path = Config.PLOTS_DIR / 'confusion_matrix.png'
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        if save_path is None:
            save_path = Config.PLOTS_DIR / 'roc_curve.png'
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        self.logger.info(f"ROC curve saved to {save_path}")
    
    def plot_training_curves(self, history=None, save_path=None):
        """Plot training and validation curves."""
        if history is None:
            if hasattr(self, 'history'):
                history = self.history
            else:
                self.logger.warning("No training history found")
                return
        
        if save_path is None:
            save_path = Config.PLOTS_DIR / 'training_curves.png'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Training curves saved to {save_path}")
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """Generate and save classification report."""
        if save_path is None:
            save_path = Config.LOGS_DIR / 'classification_report.txt'
        
        report = classification_report(y_true, y_pred,
                                      target_names=['Real', 'Fake'],
                                      output_dict=True)
        
        # Save as text
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(y_true, y_pred,
                                         target_names=['Real', 'Fake']))
        
        self.logger.info(f"Classification report saved to {save_path}")
        return report
    
    def save_metrics(self, metrics, save_path=None):
        """Save metrics to JSON file."""
        if save_path is None:
            save_path = Config.LOGS_DIR / 'evaluation_metrics.json'
        
        save_json(metrics, save_path)
        self.logger.info(f"Metrics saved to {save_path}")
    
    def run_full_evaluation(self, model_path=None):
        """Run complete evaluation pipeline."""
        # Load model
        self.load_model(model_path)
        
        # Create dataloaders
        _, _, test_loader = create_dataloaders()
        
        # Evaluate on test set
        metrics, y_true, y_pred, y_prob = self.evaluate(test_loader, "Test")
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_prob)
        
        # Load and plot training history
        if Config.TRAINING_HISTORY_PATH.exists():
            with open(Config.TRAINING_HISTORY_PATH, 'r') as f:
                history = json.load(f)
            self.plot_training_curves(history)
        
        # Generate classification report
        self.generate_classification_report(y_true, y_pred)
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Print final summary
        self.print_summary(metrics)
        
        return metrics
    
    def print_summary(self, metrics):
        """Print evaluation summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info("=" * 60)

def main():
    """Main function to run evaluation."""
    evaluator = Evaluator()
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()