"""Window for displaying shot video and feedback."""
import cv2
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QTextEdit, QWidget, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


class ShotDetailWindow(QDialog):
    """Window showing shot video and Klay's feedback."""
    
    def __init__(self, video_path: str, critique: str, shot_made: bool, shot_number: int, parent=None):
        """Initialize detail window.
        
        Args:
            video_path: Path to video file
            critique: Klay's feedback text
            shot_made: Whether shot was made
            shot_number: Shot number
            parent: Parent window
        """
        super().__init__(parent)
        self.video_path = video_path
        self.critique = critique
        self.shot_made = shot_made
        self.shot_number = shot_number
        
        self.setWindowTitle(f"Shot #{shot_number} - {'Made' if shot_made else 'Missed'}")
        self.setGeometry(100, 100, 1200, 800)
        
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        
        self.init_ui()
        self.load_video()
    
    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header_layout = QHBoxLayout()
        result_label = QLabel(f"Shot #{self.shot_number} - {'✓ Made' if self.shot_made else '✗ Missed'}")
        result_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(result_label)
        header_layout.addStretch()
        
        # Play/Pause button
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        header_layout.addWidget(self.play_btn)
        
        layout.addLayout(header_layout)
        
        # Main content - split horizontally
        content_layout = QHBoxLayout()
        
        # Left side - Video
        video_widget = QWidget()
        video_layout = QVBoxLayout()
        video_widget.setLayout(video_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        video_layout.addWidget(self.video_label)
        
        content_layout.addWidget(video_widget, 2)  # Video takes 2/3 of space
        
        # Right side - Feedback
        feedback_widget = QWidget()
        feedback_layout = QVBoxLayout()
        feedback_widget.setLayout(feedback_layout)
        feedback_widget.setMaximumWidth(400)
        
        feedback_title = QLabel("Klay's Feedback")
        feedback_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        feedback_layout.addWidget(feedback_title)
        
        # Scrollable text area for feedback
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 1px solid #ddd;")
        
        feedback_text = QTextEdit()
        feedback_text.setReadOnly(True)
        feedback_text.setPlainText(self.critique)
        feedback_text.setStyleSheet("""
            QTextEdit {
                font-size: 13px;
                line-height: 1.6;
                padding: 15px;
                background-color: #f9f9f9;
            }
        """)
        scroll.setWidget(feedback_text)
        
        feedback_layout.addWidget(scroll)
        
        content_layout.addWidget(feedback_widget, 1)  # Feedback takes 1/3 of space
        
        layout.addLayout(content_layout)
        
        # Video playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_video_frame)
    
    def load_video(self):
        """Load video file."""
        if not Path(self.video_path).exists():
            self.video_label.setText(f"Video not found:\n{self.video_path}")
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.video_label.setText(f"Failed to open video:\n{self.video_path}")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if fps > 0:
            interval = int(1000 / fps)  # Convert fps to milliseconds
            self.playback_timer.setInterval(interval)
        
        # Show first frame
        self.update_video_frame()
    
    def toggle_playback(self):
        """Toggle video playback."""
        if not self.cap or not self.cap.isOpened():
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_btn.setText("Pause")
            self.playback_timer.start()
        else:
            self.play_btn.setText("Play")
            self.playback_timer.stop()
    
    def update_video_frame(self):
        """Update video frame display."""
        if not self.cap or not self.cap.isOpened():
            return
        
        if self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                # End of video, restart
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        else:
            # Show current frame (paused)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Convert frame to QPixmap
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.cap:
            self.cap.release()
        self.playback_timer.stop()
        event.accept()

