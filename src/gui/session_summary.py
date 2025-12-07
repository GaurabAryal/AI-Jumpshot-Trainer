"""Session summary dialog."""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QTextEdit, QListWidget, QListWidgetItem,
                            QMessageBox)
from PyQt6.QtCore import Qt
from pathlib import Path


class SessionSummaryDialog(QDialog):
    """Dialog showing session summary."""
    
    def __init__(self, session_metadata: dict, parent=None):
        """Initialize session summary dialog.
        
        Args:
            session_metadata: Session metadata dictionary
            parent: Parent window
        """
        super().__init__(parent)
        self.session_metadata = session_metadata
        self.setWindowTitle("Session Summary")
        self.setGeometry(200, 200, 900, 700)
        
        self.init_ui()
        self.load_session_data()
    
    def init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Session stats
        stats_layout = QHBoxLayout()
        stats_text = (
            f"Total Shots: {self.session_metadata.get('total_shots', 0)} | "
            f"Made: {self.session_metadata.get('shots_made', 0)} | "
            f"Missed: {self.session_metadata.get('shots_missed', 0)}"
        )
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        stats_layout.addWidget(stats_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # Summary text
        summary_label = QLabel("Key Points from Klay:")
        summary_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(summary_label)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(200)
        layout.addWidget(self.summary_text)
        
        # Shot videos list
        videos_label = QLabel("Shot Videos:")
        videos_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(videos_label)
        
        self.videos_list = QListWidget()
        self.videos_list.itemDoubleClicked.connect(self.open_video)
        layout.addWidget(self.videos_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def load_session_data(self):
        """Load session data into UI."""
        # Load summary
        summary = self.session_metadata.get('summary', 'No summary available.')
        self.summary_text.setPlainText(summary)
        
        # Load shot videos
        shots = self.session_metadata.get('shots', [])
        for shot in shots:
            shot_num = shot.get('shot_number', 0)
            shot_made = shot.get('shot_made', False)
            video_path = shot.get('video_path', '')
            critique = shot.get('critique', '')
            
            result = "Made" if shot_made else "Missed"
            item_text = f"Shot {shot_num} - {result}"
            if critique:
                item_text += f": {critique[:50]}..."
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_path)
            self.videos_list.addItem(item)
    
    def open_video(self, item: QListWidgetItem):
        """Open video file.
        
        Args:
            item: List item that was double-clicked
        """
        video_path = item.data(Qt.ItemDataRole.UserRole)
        if video_path and Path(video_path).exists():
            import subprocess
            import platform
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', video_path])
            elif platform.system() == 'Windows':
                subprocess.run(['start', video_path], shell=True)
            else:  # Linux
                subprocess.run(['xdg-open', video_path])
        else:
            QMessageBox.warning(self, "File Not Found", f"Video file not found:\n{video_path}")

