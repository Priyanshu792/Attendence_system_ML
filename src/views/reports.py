import customtkinter as ctk
from datetime import datetime, timedelta
import pandas as pd
from tkinter import ttk, messagebox
import logging

from src.core.base_window import BaseWindow
from src.config.db_config import DatabaseConnection

class ReportsView(BaseWindow):
    def __init__(self, root=None):
        super().__init__(root, "Attendance Reports")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup reports interface"""
        # Filters frame
        filters_frame = ctk.CTkFrame(self.container)
        filters_frame.pack(fill="x", padx=20, pady=10)
        
        # Date range
        date_frame = ctk.CTkFrame(filters_frame)
        date_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(date_frame, text="Date Range:").pack(side="left", padx=5)
        
        self.start_date = ctk.CTkEntry(date_frame)
        self.start_date.pack(side="left", padx=5)
        self.start_date.insert(0, (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        
        ctk.CTkLabel(date_frame, text="to").pack(side="left", padx=5)  # Fixed syntax error here
        
        self.end_date = ctk.CTkEntry(date_frame)
        self.end_date.pack(side="left", padx=5)
        self.end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        # Generate button
        ctk.CTkButton(
            filters_frame,
            text="Generate Report",
            command=self.generate_report
        ).pack(side="right", padx=10)
        
        # Report area
        self.report_frame = ctk.CTkFrame(self.container)
        self.report_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create treeview for report data
        self.tree = ttk.Treeview(
            self.report_frame,
            columns=("student_id", "name", "total_days", "present", "percentage")
        )
        
        # Configure columns
        self.tree.heading("student_id", text="Student ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("total_days", text="Total Days")
        self.tree.heading("present", text="Days Present")
        self.tree.heading("percentage", text="Attendance %")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            self.report_frame,
            orient="vertical",
            command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def generate_report(self):
        """Generate attendance report"""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            with DatabaseConnection() as cursor:
                cursor.execute("""
                    SELECT 
                        s.student_id,
                        s.name,
                        COUNT(DISTINCT a.date) as days_present,
                        (julianday(?) - julianday(?)) as total_days
                    FROM students s
                    LEFT JOIN attendance a ON s.student_id = a.student_id
                    AND a.date BETWEEN ? AND ?
                    GROUP BY s.student_id, s.name
                """, (
                    self.end_date.get(),
                    self.start_date.get(),
                    self.start_date.get(),
                    self.end_date.get()
                ))
                
                for row in cursor.fetchall():
                    student_id, name, days_present, total_days = row
                    total_days = int(total_days) + 1
                    attendance_pct = (days_present / total_days) * 100 if total_days > 0 else 0
                    
                    self.tree.insert("", "end", values=(
                        student_id,
                        name,
                        total_days,
                        days_present,
                        f"{attendance_pct:.1f}%"
                    ))
                    
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Could not generate report: {str(e)}")
