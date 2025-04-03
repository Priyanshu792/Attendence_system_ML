import tkinter as tk
import customtkinter as ctk
from typing import Optional

class BaseWindow:
    """Base window class with common functionality"""
    
    def __init__(self, root: Optional[tk.Misc] = None, title: str = "Window"):
        if root is None or isinstance(root, tk.Tk):
            self.root = root or tk.Tk()
            self.root.title(title)
            self.setup_window()
            self.container = ctk.CTkFrame(self.root)
        else:
            # If root is a frame, use it directly as container
            self.root = root
            self.container = root
            
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        
    def setup_window(self):
        """Configure window properties"""
        if isinstance(self.root, tk.Tk):
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Set window size to 80% of screen size
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)
            
            # Center window
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.root.resizable(True, True)
