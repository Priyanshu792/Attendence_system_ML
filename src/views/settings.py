import customtkinter as ctk
import json
import os
from pathlib import Path
from tkinter import messagebox
import logging

from src.core.base_window import BaseWindow

class SettingsView(BaseWindow):
    def __init__(self, root=None):
        super().__init__(root, "Settings")
        self.settings_file = Path(__file__).parent.parent.parent / "config" / "settings.json"
        self.load_settings()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup settings interface"""
        # Create settings sections
        sections = [
            ("General", self.create_general_section),
            ("Recognition", self.create_recognition_section),
            ("Database", self.create_database_section)
        ]
        
        # Create tabs
        self.tabview = ctk.CTkTabview(self.container)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Add tabs
        for section_name, create_func in sections:
            tab = self.tabview.add(section_name)
            create_func(tab)
            
        # Add save button
        ctk.CTkButton(
            self.container,
            text="Save Settings",
            command=self.save_settings
        ).pack(pady=10)

    def create_general_section(self, parent):
        """Create general settings section"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Theme selection
        ctk.CTkLabel(frame, text="Theme:").pack(anchor="w", padx=5, pady=5)
        self.theme_var = ctk.StringVar(value=self.settings.get("theme", "system"))
        theme_menu = ctk.CTkOptionMenu(
            frame,
            values=["light", "dark", "system"],
            variable=self.theme_var
        )
        theme_menu.pack(anchor="w", padx=5, pady=5)
        
        # Language selection
        ctk.CTkLabel(frame, text="Language:").pack(anchor="w", padx=5, pady=5)
        self.lang_var = ctk.StringVar(value=self.settings.get("language", "en"))
        lang_menu = ctk.CTkOptionMenu(
            frame,
            values=["en", "es", "fr"],
            variable=self.lang_var
        )
        lang_menu.pack(anchor="w", padx=5, pady=5)

    def create_recognition_section(self, parent):
        """Create recognition settings section"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Confidence threshold
        ctk.CTkLabel(frame, text="Confidence Threshold:").pack(anchor="w", padx=5, pady=5)
        self.confidence_var = ctk.StringVar(value=str(self.settings.get("confidence_threshold", 85)))
        confidence_entry = ctk.CTkEntry(frame, textvariable=self.confidence_var)
        confidence_entry.pack(anchor="w", padx=5, pady=5)
        
        # Face detection settings
        ctk.CTkLabel(frame, text="Min Face Size:").pack(anchor="w", padx=5, pady=5)
        self.min_face_var = ctk.StringVar(value=str(self.settings.get("min_face_size", 30)))
        min_face_entry = ctk.CTkEntry(frame, textvariable=self.min_face_var)
        min_face_entry.pack(anchor="w", padx=5, pady=5)

    def create_database_section(self, parent):
        """Create database settings section"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Database path
        ctk.CTkLabel(frame, text="Database Path:").pack(anchor="w", padx=5, pady=5)
        self.db_path_var = ctk.StringVar(value=self.settings.get("db_path", "data/face_recognition.db"))
        db_path_entry = ctk.CTkEntry(frame, textvariable=self.db_path_var)
        db_path_entry.pack(anchor="w", padx=5, pady=5)

    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
            else:
                self.settings = {}
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            self.settings = {}

    def save_settings(self):
        """Save settings to file"""
        try:
            settings = {
                "theme": self.theme_var.get(),
                "language": self.lang_var.get(),
                "confidence_threshold": float(self.confidence_var.get()),
                "min_face_size": int(self.min_face_var.get()),
                "db_path": self.db_path_var.get()
            }
            
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
                
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Could not save settings: {str(e)}")
