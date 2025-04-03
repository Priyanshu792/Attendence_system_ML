import sqlite3
import logging
from pathlib import Path
from typing import List, Dict

class Migration:
    """Base migration class"""
    version: int
    up_sql: str
    down_sql: str

class InitialMigration(Migration):
    version = 1
    up_sql = '''
    CREATE TABLE students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE,
        name TEXT NOT NULL,
        email TEXT,
        course TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        date DATE,
        time TIME,
        status TEXT,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );

    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    '''
    down_sql = '''
    DROP TABLE IF EXISTS attendance;
    DROP TABLE IF EXISTS students;
    DROP TABLE IF EXISTS settings;
    '''

def get_migrations() -> List[Migration]:
    """Get all migrations in order"""
    return [
        InitialMigration
    ]

def migrate(db_path: str):
    """Run all pending migrations"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create migrations table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Get applied migrations
        cursor.execute("SELECT version FROM migrations")
        applied = {row[0] for row in cursor.fetchall()}

        # Apply pending migrations
        for migration in get_migrations():
            if migration.version not in applied:
                logging.info(f"Applying migration {migration.version}")
                cursor.executescript(migration.up_sql)
                cursor.execute(
                    "INSERT INTO migrations (version) VALUES (?)",
                    (migration.version,)
                )

        conn.commit()
    except Exception as e:
        logging.error(f"Migration error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
