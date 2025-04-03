import customtkinter as ctk
from datetime import datetime
import pandas as pd
from tkinter import ttk, messagebox
import logging

from src.core.base_window import BaseWindow
from src.config.db_config import DatabaseConnection

class AttendanceView(BaseWindow):
    def __init__(self, root=None):
        super().__init__(root, "Attendance Management")
        self.setup_ui()
        self.load_attendance()

    def setup_ui(self):
        # Create toolbar
        toolbar = ctk.CTkFrame(self.container)
        toolbar.pack(fill="x", padx=10, pady=5)

        # Date picker
        self.date_var = ctk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        date_entry = ctk.CTkEntry(toolbar, textvariable=self.date_var)
        date_entry.pack(side="left", padx=5)

        # Refresh button
        ctk.CTkButton(
            toolbar,
            text="Refresh",
            command=self.load_attendance
        ).pack(side="left", padx=5)

        # Export button
        ctk.CTkButton(
            toolbar,
            text="Export CSV",
            command=self.export_csv
        ).pack(side="right", padx=5)

        # Create treeview
        self.tree = ttk.Treeview(self.container, columns=(
            "student_id", "name", "time", "date", "status"
        ))

        # Configure columns
        self.tree.heading("student_id", text="Student ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("time", text="Time")
        self.tree.heading("date", text="Date")
        self.tree.heading("status", text="Status")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Pack elements
        self.tree.pack(fill="both", expand=True, padx=10, pady=5)
        scrollbar.pack(side="right", fill="y")

    def load_attendance(self):
        """Load attendance data from database"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            with DatabaseConnection() as cursor:
                cursor.execute("""
                    SELECT a.student_id, s.name, a.time, a.date, a.status
                    FROM attendance a
                    JOIN students s ON a.student_id = s.student_id
                    WHERE a.date = ?
                    ORDER BY a.time DESC
                """, (self.date_var.get(),))
                
                for row in cursor.fetchall():
                    self.tree.insert("", "end", values=row)

        except Exception as e:
            logging.error(f"Error loading attendance: {e}")
            messagebox.showerror("Error", f"Could not load attendance: {str(e)}")

    def export_csv(self):
        """Export attendance data to CSV"""
        try:
            data = []
            for item in self.tree.get_children():
                data.append(self.tree.item(item)["values"])

            df = pd.DataFrame(
                data,
                columns=["Student ID", "Name", "Time", "Date", "Status"]
            )

            filename = f"attendance_{self.date_var.get()}.csv"
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Exported to {filename}")

        except Exception as e:
            logging.error(f"Error exporting CSV: {e}")
            messagebox.showerror("Error", f"Could not export CSV: {str(e)}")
