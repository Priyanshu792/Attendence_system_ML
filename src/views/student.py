import customtkinter as ctk
from typing import Optional
import cv2
from PIL import Image, ImageTk
import logging
from tkinter import messagebox
import os
from pathlib import Path

from src.core.base_window import BaseWindow
from src.utils.face_utils import FaceDetector
from src.config.db_config import DatabaseConnection

class StudentView(BaseWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_detector = FaceDetector()
        self.cap = None
        self.is_capturing = False
        self.capture_count = 0
        self.max_captures = 100  # Changed to 100 images
        self._current_image = None  # Add this line
        self.setup_ui()
        self.container.bind("<Destroy>", lambda e: self.cleanup())

    def setup_ui(self):
        # Create form frame
        form_frame = ctk.CTkFrame(self.container)
        form_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Student details form
        fields = [
            ("Student ID", "student_id"),
            ("Name", "name"),
            ("Course", "course"),
            ("Year", "year"),
            ("Semester", "semester"),
            ("Email", "email")
        ]

        self.entries = {}
        for label, field in fields:
            frame = ctk.CTkFrame(form_frame)
            frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(frame, text=label).pack(side="left")
            self.entries[field] = ctk.CTkEntry(frame)
            self.entries[field].pack(side="right", expand=True, fill="x", padx=5)

        # Buttons
        btn_frame = ctk.CTkFrame(form_frame)
        btn_frame.pack(fill="x", pady=10)

        # Single save button that starts the whole process
        self.save_btn = ctk.CTkButton(
            btn_frame,
            text="Save & Capture Photos",
            command=self.save_and_capture
        )
        self.save_btn.pack(side="right", padx=5)

        # Progress indicators
        self.status_label = ctk.CTkLabel(
            form_frame,
            text="Fill in details and click Save & Capture"
        )
        self.status_label.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(form_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.set(0)

        # Camera frame
        self.camera_frame = ctk.CTkFrame(self.container)
        self.camera_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Update camera_label creation
        self.camera_label = ctk.CTkLabel(
            self.camera_frame, 
            text="Camera Feed",
            width=640,  # Add fixed size
            height=480
        )
        self.camera_label.pack(expand=True)

    def save_and_capture(self):
        """Save student details and start automatic photo capture"""
        # Validate form data
        student_data = {field: entry.get() for field, entry in self.entries.items()}
        if not all([student_data["student_id"], student_data["name"]]):
            messagebox.showerror("Error", "Student ID and Name are required")
            return

        try:
            # Save to database first
            with DatabaseConnection() as cursor:
                cursor.execute("""
                    INSERT INTO students 
                    (student_id, name, email, course)
                    VALUES (?, ?, ?, ?)
                """, (
                    student_data["student_id"],
                    student_data["name"],
                    student_data["email"],
                    student_data["course"]
                ))
            
            # Start camera capture
            self.save_btn.configure(state="disabled")
            self.status_label.configure(text="Starting camera...")
            self.cap = cv2.VideoCapture(0)
            self.is_capturing = True
            self.capture_count = 0
            self.progress_bar.set(0)
            self.container.after(1000, self.auto_capture)  # Start after 1 second delay
            
        except Exception as e:
            logging.error(f"Error saving student: {e}")
            messagebox.showerror("Error", f"Could not save student: {str(e)}")

    def auto_capture(self):
        """Automatically capture photos"""
        if not self.is_capturing or self.capture_count >= self.max_captures:
            self.cleanup()
            self.status_label.configure(text="Photo capture completed")
            self.save_btn.configure(state="normal")
            return

        ret, frame = self.cap.read()
        if ret:
            faces = self.face_detector.detect_faces(frame)
            if len(faces) == 1:  # Only capture if exactly one face is detected
                # Save the face image
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]
                
                # Create directory for training data
                data_dir = Path(__file__).parent.parent.parent / "data" / "training_images"
                data_dir.mkdir(parents=True, exist_ok=True)

                # Save face image
                student_id = self.entries["student_id"].get()
                filename = f"{data_dir}/user.{student_id}.{self.capture_count}.jpg"
                cv2.imwrite(str(filename), face)

                self.capture_count += 1
                progress = self.capture_count / self.max_captures
                self.progress_bar.set(progress)
                self.status_label.configure(
                    text=f"Capturing photos: {self.capture_count}/{self.max_captures}"
                )

            # Update display with proper image handling
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            # Resize image to fit label
            pil_img = pil_img.resize((640, 480), Image.Resampling.LANCZOS)
            self._current_image = ctk.CTkImage(light_image=pil_img, size=(640, 480))
            self.camera_label.configure(image=self._current_image)

        # Schedule next capture
        if self.is_capturing:
            self.container.after(100, self.auto_capture)  # Capture every 100ms

    def cleanup(self):
        """Cleanup resources"""
        self.is_capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
