"""
Prediction module for NAV-SMFS.
Single image prediction with confidence score.
Uses the global ModelLoader for efficient inference.
"""

import sys
from pathlib import Path

# Import the model loader
from model_loader import predict_image as model_predict

def predict_image(image_path):
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
    return model_predict(image_path)

def main():
    """Command-line interface for prediction."""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    try:
        # Make prediction
        result = predict_image(image_path)
        
        # Print results
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Image: {image_path}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("-" * 50)
        print("Class Probabilities:")
        print(f"  Real: {result['real_probability']:.2f}%")
        print(f"  Fake: {result['fake_probability']:.2f}%")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()