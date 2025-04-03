import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
import os
import urllib.request
from pathlib import Path

class FaceDetector:
    CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = str(Path(__file__).parent.parent.parent / "data" / "haarcascade_frontalface_default.xml")
        
        # Download cascade file if it doesn't exist
        if not os.path.exists(cascade_path):
            os.makedirs(os.path.dirname(cascade_path), exist_ok=True)
            try:
                urllib.request.urlretrieve(self.CASCADE_URL, cascade_path)
                logging.info(f"Downloaded cascade file to {cascade_path}")
            except Exception as e:
                logging.error(f"Error downloading cascade file: {e}")
                raise
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_loaded = False

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame and return list of (x,y,w,h) coordinates"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,        # More gradual scaling
            minNeighbors=5,         # Reduced to detect more faces
            minSize=(30, 30),       # Smaller minimum size
            maxSize=(300, 300),     # Add maximum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def predict_face(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[int, float]:
        """Predict identity of face in frame"""
        if not self.model_loaded:
            return -1, 0.0
            
        x, y, w, h = face_coords
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        
        # Normalize ROI
        roi = cv2.equalizeHist(roi)
        roi = cv2.resize(roi, (200, 200))
        
        try:
            id_, confidence = self.recognizer.predict(roi)
            # Confidence is 0-100 where lower is better in OpenCV
            confidence = 100 - min(100, confidence)
            return int(id_), confidence
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return -1, 0.0

    def train_recognizer(self, faces: List[np.ndarray], labels: List[int]):
        """Train the face recognizer"""
        processed_faces = []
        for face in faces:
            if face is not None:
                face = cv2.equalizeHist(face)
                face = cv2.resize(face, (200, 200))
                processed_faces.append(face)
            
        if not processed_faces:
            raise ValueError("No valid faces for training")
            
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,           # Smaller radius for more detail
            neighbors=8,        # Standard number of points
            grid_x=8,          # Standard grid
            grid_y=8,          # Standard grid
            threshold=100.0     # Higher threshold for better matching
        )
        self.recognizer.train(processed_faces, np.array(labels))
        self.model_loaded = True
        
    def save_trained_model(self, path: str = None):
        """Save trained model to file"""
        if path is None:
            path = str(Path(__file__).parent.parent.parent / "data" / "models" / "classifier.xml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.recognizer.save(path)
        return path

    def load_trained_model(self, path: str = None):
        """Load trained model from file"""
        if path is None:
            path = str(Path(__file__).parent.parent.parent / "data" / "models" / "classifier.xml")
        try:
            if os.path.exists(path):
                self.recognizer.read(path)
                self.model_loaded = True
            else:
                logging.warning(f"Model file not found: {path}")
                self.model_loaded = False
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.model_loaded = False
