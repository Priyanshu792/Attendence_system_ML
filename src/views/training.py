import customtkinter as ctk
import cv2
import os
import numpy as np
from PIL import Image
import logging
from tkinter import messagebox
from pathlib import Path
from collections import Counter
import random

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
        # Create popup window
        popup = ctk.CTkToplevel()
        popup.title("Training Results")
        popup.geometry("800x600")
        
        # Keep reference to prevent garbage collection
        self._popup_images = []

        # Create main frame
        main_frame = ctk.CTkFrame(popup)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Statistics section
        stats_frame = ctk.CTkFrame(main_frame)
        stats_frame.pack(fill="x", padx=10, pady=10)

        try:
            # Load data statistics
            data_dir = Path(__file__).parent.parent.parent / "data" / "training_images"
            if not data_dir.exists():
                raise ValueError("No training data found")

            # Collect statistics
            files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            student_ids = [int(f.split('.')[1]) for f in files if len(f.split('.')) > 2]
            id_counts = Counter(student_ids)

            # Display statistics
            stats_text = f"""
            Training Dataset Statistics:
            - Total images: {len(files)}
            - Unique students: {len(id_counts)}
            - Images per student (avg): {len(files)/len(id_counts):.1f}
            - Model status: {"Trained" if self.face_detector.model_loaded else "Not trained"}
            """
            
            ctk.CTkLabel(
                stats_frame,
                text=stats_text,
                justify="left",
                font=("Helvetica", 12)
            ).pack(padx=10, pady=10)

            # Sample images section
            samples_frame = ctk.CTkFrame(main_frame)
            samples_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            ctk.CTkLabel(
                samples_frame,
                text="Sample Training Images:",
                font=("Helvetica", 12, "bold")
            ).pack(pady=5)

            # Display grid of sample images
            grid_frame = ctk.CTkFrame(samples_frame)
            grid_frame.pack(fill="both", expand=True, padx=5, pady=5)

            # Randomly select up to 6 sample images
            sample_files = random.sample(files, min(6, len(files)))
            
            for i, file in enumerate(sample_files):
                row = i // 3
                col = i % 3
                
                # Load and display image
                img_path = data_dir / file
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (160, 160))
                pil_img = Image.fromarray(img)
                photo = ctk.CTkImage(light_image=pil_img, size=(160, 160))
                # Keep reference to image
                self._popup_images.append(photo)
                
                # Create frame for each image with caption
                img_frame = ctk.CTkFrame(grid_frame)
                img_frame.grid(row=row, column=col, padx=5, pady=5)
                
                ctk.CTkLabel(
                    img_frame,
                    image=photo,
                    text=""
                ).pack(padx=5, pady=2)
                
                student_id = file.split('.')[1]
                ctk.CTkLabel(
                    img_frame,
                    text=f"Student ID: {student_id}",
                    font=("Helvetica", 10)
                ).pack(pady=2)

        except Exception as e:
            logging.error(f"Error showing results: {e}")
            ctk.CTkLabel(
                stats_frame,
                text=f"Error loading results: {str(e)}",
                text_color="red"
            ).pack(padx=10, pady=10)

        # Center popup window
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")
