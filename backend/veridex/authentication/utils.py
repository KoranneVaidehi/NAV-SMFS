import cv2
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image
import tempfile

def is_image_file(filename):
    """
    Check if file is an image based on extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if image, False otherwise
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions

def is_video_file(filename):
    """
    Check if file is a video based on extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if video, False otherwise
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in video_extensions

def get_file_extension(filename):
    """
    Get file extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: File extension with dot
    """
    return os.path.splitext(filename)[1].lower()

def detect_faces(image_path, detector):
    """
    Detect faces in an image using MTCNN.
    
    Args:
        image_path: Path to the image file
        detector: MTCNN detector instance
        
    Returns:
        list: List of face dictionaries with bounding boxes and face images
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at {image_path}")
            return []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        
        # Process each face
        face_data = []
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                # Crop face
                face_roi = image_rgb[y:y+h, x:x+w]
                
                # Convert to base64 for display
                if face_roi.size > 0:
                    face_pil = Image.fromarray(face_roi)
                    buffered = BytesIO()
                    face_pil.save(buffered, format="JPEG")
                    face_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    face_data.append({
                        'box': face['box'],
                        'confidence': face['confidence'],
                        'keypoints': face.get('keypoints', {}),
                        'face_image': face_base64
                    })
        
        return face_data
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def detect_faces_video(video_path, detector, frame_interval=30):
    """
    Detect faces in a video file.
    
    Args:
        video_path: Path to the video file
        detector: MTCNN detector instance
        frame_interval: Process every Nth frame
        
    Returns:
        tuple: (faces_count, faces_data)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video at {video_path}")
            return 0, []
        
        faces_count = 0
        face_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(frame_rgb)
                faces_count += len(faces)
                
                # Store a sample face if found
                if faces and len(face_frames) < 5:
                    x, y, w, h = faces[0]['box']
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    if w > 0 and h > 0:
                        face_roi = frame_rgb[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            face_pil = Image.fromarray(face_roi)
                            buffered = BytesIO()
                            face_pil.save(buffered, format="JPEG")
                            face_base64 = base64.b64encode(buffered.getvalue()).decode()
                            face_frames.append({
                                'frame': frame_count,
                                'face_image': face_base64,
                                'box': [x, y, w, h]
                            })
            
            frame_count += 1
        
        cap.release()
        return faces_count, face_frames
    except Exception as e:
        print(f"Error in video face detection: {e}")
        return 0, []

def draw_faces_on_image(image, faces):
    """
    Draw bounding boxes around detected faces.
    
    Args:
        image: cv2 image
        faces: List of face dictionaries
        
    Returns:
        cv2 image with bounding boxes drawn
    """
    img_copy = image.copy()
    
    for face in faces:
        x, y, w, h = face['box']
        
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w > 0 and h > 0:
            # Draw rectangle
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence score
            confidence = face.get('confidence', 0)
            cv2.putText(img_copy, f"{confidence:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints if available
            if 'keypoints' in face and face['keypoints']:
                keypoints = face['keypoints']
                for keypoint in keypoints.values():
                    if isinstance(keypoint, (list, tuple)) and len(keypoint) == 2:
                        cv2.circle(img_copy, (keypoint[0], keypoint[1]), 3, (255, 0, 0), -1)
    
    return img_copy

def save_uploaded_file(file, destination):
    """
    Save an uploaded file to destination.
    
    Args:
        file: Uploaded file object
        destination: Path to save the file
        
    Returns:
        str: Path to the saved file
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb+') as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    return str(destination)

def get_file_size(file_path):
    """
    Get file size in human-readable format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File size with unit
    """
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "Unknown"

def validate_image(image_path, max_size_mb=20):
    """
    Validate an image file.
    
    Args:
        image_path: Path to the image
        max_size_mb: Maximum file size in MB
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "File does not exist"
        
        # Check file size
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File size exceeds {max_size_mb}MB limit"
        
        # Check if it's a valid image
        img = Image.open(image_path)
        img.verify()
        
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"