import customtkinter as ctk
import cv2
import os
import numpy as np
from PIL import Image
import logging
from tkinter import messagebox
from pathlib import Path

from src.core.base_window import BaseWindow
from src.utils.face_utils import FaceDetector

class TrainingView(BaseWindow):
    def __init__(self, root=None):
        super().__init__(root, "Model Training")
        self.face_detector = FaceDetector()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup training interface"""
        # Status section
        status_frame = ctk.CTkFrame(self.container)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready to train model",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(status_frame)
        self.progress.pack(fill="x", padx=20, pady=10)
        self.progress.set(0)
        
        # Controls
        btn_frame = ctk.CTkFrame(self.container)
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="Start Training",
            command=self.start_training
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame,
            text="View Results",
            command=self.view_results
        ).pack(side="left", padx=10)

    def start_training(self):
        """Start the training process"""
        try:
            self.status_label.configure(text="Loading training data...")
            self.progress.set(0.2)
            
            # Load face data
            faces, ids = self._load_training_data()
            
            if not faces:
                raise ValueError("No training data found")
                
            self.status_label.configure(text="Training model...")
            self.progress.set(0.6)
            
            # Train model
            self.face_detector.train_recognizer(faces, ids)
            
            self.status_label.configure(text="Saving model...")
            self.progress.set(0.8)
            
            # Save model
            self.face_detector.save_trained_model()
            
            self.status_label.configure(text="Training completed successfully")
            self.progress.set(1.0)
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            self.status_label.configure(text=f"Error: {str(e)}")
            self.progress.set(0)

    def _load_training_data(self):
        """Load training images and prepare data"""
        faces = []
        ids = []
        
        data_dir = Path(__file__).parent.parent.parent / "data" / "training_images"
        if not data_dir.exists():
            raise ValueError("No training data directory found")
            
        for image_file in os.listdir(data_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(data_dir, image_file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                try:
                    id_num = int(image_file.split('.')[1])
                    faces.append(img)
                    ids.append(id_num)
                except:
                    logging.warning(f"Skipping invalid filename: {image_file}")
                    
        if not faces:
            raise ValueError("No valid training images found")
                    
        return faces, ids

    def save_trained_model(self):
        """Save trained model to file"""
        models_dir = Path(__file__).parent.parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / "classifier.xml"
        self.face_detector.save_trained_model(str(path))
        messagebox.showinfo("Success", "Model saved successfully")

    def view_results(self):
        """Show training results and statistics"""
        # Add visualization of training results
        pass
