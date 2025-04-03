import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

class ConfigManager:
    """Manages application configuration"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config_file = Path(__file__).parent / "config.json"
            self.config: Dict[str, Any] = {}
            self.load_config()
            self.initialized = True
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            self.config = self._get_default_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "database": {
                "path": "data/face_recognition.db"
            },
            "face_detection": {
                "confidence_threshold": 85,
                "min_face_size": 30,
                "scale_factor": 1.1
            },
            "training": {
                "batch_size": 32,
                "epochs": 10,
                "validation_split": 0.2
            },
            "ui": {
                "theme": "system",
                "language": "en"
            }
        }
