import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import logging
from pathlib import Path
from typing import Dict, Any

from src.core.base_window import BaseWindow
from src.core.theme_manager import ThemeManager
from src.config.db_config import init_database
from src.views.student import StudentView
from src.views.recognition import RecognitionView
from src.views.attendance import AttendanceView
from src.views.training import TrainingView
from src.views.reports import ReportsView
from src.views.settings import SettingsView

# Configure logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class ModernFaceRecognition(BaseWindow):
    """Modern Face Recognition Attendance System"""
    
    def __init__(self):
        super().__init__(title="Modern Face Recognition System")
        self.theme = ThemeManager()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the modern UI"""
        # Create sidebar
        self.sidebar = ctk.CTkFrame(
            self.container,
            width=200,
            corner_radius=0
        )
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        
        # Create main content area
        self.content = ctk.CTkFrame(self.container)
        self.content.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.create_sidebar_buttons()
        self.show_welcome_screen()
    
    def create_sidebar_buttons(self):
        """Create sidebar navigation buttons"""
        buttons = [
            ("Students", self.show_students),
            ("Face Recognition", self.show_face_recognition),
            ("Attendance", self.show_attendance),
            ("Training", self.show_training),
            ("Reports", self.show_reports),
            ("Settings", self.show_settings)
        ]
        
        for text, command in buttons:
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                **self.theme.get_style("button")
            )
            btn.pack(pady=5, padx=10, fill="x")
    
    def show_welcome_screen(self):
        """Show welcome screen in content area"""
        # Clear existing content
        for widget in self.content.winfo_children():
            widget.destroy()
            
        welcome = ctk.CTkLabel(
            self.content,
            text="Face Recognition System",
            font=("Helvetica", 24, "bold")
        )
        welcome.pack(pady=20)

    def show_students(self):
        """Show student management view"""
        for widget in self.content.winfo_children():
            widget.destroy()
        StudentView(self.content)

    def show_face_recognition(self):
        """Show face recognition view"""
        for widget in self.content.winfo_children():
            widget.destroy()
        RecognitionView(self.content)

    def show_attendance(self):
        """Show attendance management view"""
        for widget in self.content.winfo_children():
            widget.destroy()
        AttendanceView(self.content)

    def show_training(self):
        """Show model training view"""
        for widget in self.content.winfo_children():
            widget.destroy()
        TrainingView(self.content)

    def show_reports(self):
        """Show reports view"""
        for widget in self.content.winfo_children():
            widget.destroy()
        ReportsView(self.content)

    def show_settings(self):
        """Show settings view"""
        for widget in self.content.winfo_children():
            widget.destroy()
        SettingsView(self.content)

def main():
    try:
        # Initialize database
        init_database()
        
        # Start application
        app = ModernFaceRecognition()
        app.root.mainloop()
    except Exception as e:
        logging.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()
