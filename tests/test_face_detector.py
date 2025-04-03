import unittest
import numpy as np
import cv2
from pathlib import Path
from src.utils.face_utils import FaceDetector
from src.utils.image_utils import enhance_image, normalize_face

class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
    def test_face_detection(self):
        # Create a simple test image with a face-like pattern
        cv2.circle(self.test_image, (150, 150), 50, (255, 255, 255), -1)
        
        faces = self.detector.detect_faces(self.test_image)
        self.assertGreater(len(faces), 0, "Should detect at least one face")
        
    def test_face_prediction(self):
        # Test with known face image
        face_coords = (100, 100, 100, 100)
        id_, confidence = self.detector.predict_face(self.test_image, face_coords)
        
        self.assertIsInstance(id_, int)
        self.assertIsInstance(confidence, float)
        
    def test_image_enhancement(self):
        enhanced = enhance_image(self.test_image)
        self.assertEqual(len(enhanced.shape), 2, "Should be grayscale")
        
    def test_face_normalization(self):
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        normalized = normalize_face(face)
        
        self.assertEqual(normalized.shape, (160, 160))
        self.assertAlmostEqual(normalized.mean(), 0, places=5)
