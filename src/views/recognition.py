import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from datetime import datetime
import logging
from tkinter import messagebox

from src.core.base_window import BaseWindow
from src.utils.face_utils import FaceDetector
from src.config.db_config import DatabaseConnection

class RecognitionView(BaseWindow):
    def __init__(self, root=None):
        super().__init__(root, "Face Recognition")
        self.face_detector = FaceDetector()
        self.face_detector.load_trained_model()
        if not self.face_detector.model_loaded:
            messagebox.showwarning("Warning", "No trained model found. Please train the model first.")
        self.cap = None
        self.is_recognizing = False
        self._current_image = None
        self.setup_ui()
        self.container.bind("<Destroy>", lambda e: self.cleanup())

    def setup_ui(self):
        """Setup recognition interface"""
        # Control panel
        control_panel = ctk.CTkFrame(self.container)
        control_panel.pack(side="left", fill="y", padx=10, pady=10)
        
        # Status display
        self.status_label = ctk.CTkLabel(
            control_panel,
            text="Ready",
            font=("Helvetica", 14)
        )
        self.status_label.pack(pady=10)
        
        # Control buttons
        self.start_btn = ctk.CTkButton(
            control_panel,
            text="Start Recognition",
            command=self.start_recognition
        )
        self.start_btn.pack(pady=5)
        
        self.stop_btn = ctk.CTkButton(
            control_panel,
            text="Stop",
            command=self.stop_recognition,
            state="disabled"
        )
        self.stop_btn.pack(pady=5)
        
        # Video frame
        self.video_frame = ctk.CTkFrame(self.container)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Camera Feed",
            width=640,
            height=480
        )
        self.video_label.pack(expand=True)

    def start_recognition(self):
        """Start face recognition"""
        if not self.is_recognizing:
            self.cap = cv2.VideoCapture(0)
            self.is_recognizing = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.update_video_feed()

    def stop_recognition(self):
        """Stop face recognition"""
        self.is_recognizing = False
        if self.cap:
            self.cap.release()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Ready")

    def update_video_feed(self):
        """Update video feed and perform recognition"""
        if self.is_recognizing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                for face_coords in faces:
                    # Get predictions
                    student_id, confidence = self.face_detector.predict_face(frame, face_coords)
                    
                    if confidence > 85:  # Confidence threshold
                        try:
                            # Get student info from database
                            with DatabaseConnection() as cursor:
                                cursor.execute(
                                    "SELECT name FROM students WHERE student_id=?", 
                                    (student_id,)
                                )
                                result = cursor.fetchone()
                                name = result[0] if result else "Unknown"
                                
                                # Mark attendance
                                self.mark_attendance(student_id)
                                
                                # Draw on frame
                                x, y, w, h = face_coords
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                          (0, 255, 0), 2)
                                
                        except Exception as e:
                            logging.error(f"Database error: {e}")
                    
                # Convert to CTkImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((640, 480), Image.Resampling.LANCZOS)
                self._current_image = ctk.CTkImage(light_image=pil_img, size=(640, 480))
                self.video_label.configure(image=self._current_image)
                
            if self.is_recognizing:  # Check if still recognizing before scheduling next update
                self.container.after(10, self.update_video_feed)

    def mark_attendance(self, student_id):
        """Record attendance in database"""
        try:
            with DatabaseConnection() as cursor:
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO attendance (student_id, date, time, status)
                    VALUES (?, ?, ?, ?)
                """, (
                    student_id,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "Present"
                ))
            
        except Exception as e:
            logging.error(f"Error marking attendance: {e}")
            self.status_label.configure(text="Error marking attendance")

    def cleanup(self):
        """Cleanup resources"""
        self.is_recognizing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
