import sqlite3
from typing import Optional
import logging
from pathlib import Path

class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(__file__).parent.parent.parent / "data" / "face_recognition.db")
        self.connection = None
        self.cursor = None
        # Create data directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def __enter__(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            return self.cursor
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            if exc_type is None:
                self.connection.commit()
            self.cursor.close()
            self.connection.close()

def init_database():
    """Initialize database with required tables"""
    try:
        with DatabaseConnection() as cursor:
            # Create students table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE,
                name TEXT,
                email TEXT,
                course TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create attendance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                date DATE,
                time TIME,
                status TEXT,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
            ''')
            logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise
