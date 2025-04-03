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
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def predict_face(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Tuple[int, float]:
        """Predict identity of face in frame"""
        if not self.model_loaded:
            return -1, 0.0
        x, y, w, h = face_coords
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        
        try:
            id_, confidence = self.recognizer.predict(roi)
            return id_, confidence
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return -1, 0.0

    def train_recognizer(self, faces: List[np.ndarray], labels: List[int]):
        """Train the face recognizer"""
        self.recognizer.train(faces, np.array(labels))
        
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
