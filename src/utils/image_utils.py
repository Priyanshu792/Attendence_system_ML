import cv2
import numpy as np
from typing import Tuple, Optional
import logging

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better face detection"""
    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray

def normalize_face(face_img: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
    """Normalize face image for consistent recognition"""
    try:
        # Resize to target size
        face_img = cv2.resize(face_img, target_size)
        
        # Convert to float and normalize
        face_img = face_img.astype(np.float32)
        face_img = (face_img - face_img.mean()) / face_img.std()
        
        return face_img
    except Exception as e:
        logging.error(f"Face normalization error: {e}")
        return None

def draw_face_box(image: np.ndarray, 
                 bbox: Tuple[int, int, int, int], 
                 name: str = "", 
                 confidence: float = 0.0) -> np.ndarray:
    """Draw face detection box with name and confidence"""
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add text
    if name:
        text = f"{name} ({confidence:.1f}%)"
        cv2.putText(image, text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   (0, 255, 0), 2)
    
    return image
