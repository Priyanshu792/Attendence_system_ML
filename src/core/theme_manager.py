from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ThemeConfig:
    """Theme configuration"""
    primary_color: str = "#1a73e8"
    secondary_color: str = "#f8f9fa"
    text_color: str = "#202124"
    accent_color: str = "#ea4335"
    
    button_style: Dict[str, Any] = None
    label_style: Dict[str, Any] = None
    
    def __post_init__(self):
        self.button_style = {
            "fg_color": self.primary_color,
            "hover_color": self.accent_color,
            "text_color": "white",
            "corner_radius": 8
        }
        
        self.label_style = {
            "text_color": self.text_color,
            "fg_color": "transparent"
        }

class ThemeManager:
    """Manages application theming"""
    
    def __init__(self):
        self.current_theme = ThemeConfig()
    
    def get_style(self, element_type: str) -> Dict[str, Any]:
        """Get style configuration for an element type"""
        return getattr(self.current_theme, f"{element_type}_style")
