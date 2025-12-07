"""Main application window."""
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QCheckBox, QMessageBox, QFileDialog, QSlider,
                            QListWidget, QSplitter, QListWidgetItem, QDialog,
                            QTextEdit, QScrollArea)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from ..video.camera_manager import CameraManager
from ..video.video_file_manager import VideoFileManager
from ..video.video_recorder import VideoRecorder
from ..video.shot_detector import ShotDetector
from ..vision.pose_detector import PoseDetector
from ..session.session_manager import SessionManager
from ..session.shot_analyzer import ShotAnalyzer
from ..ai.critique_generator import CritiqueGenerator
from ..ai.gemini_client import GeminiClient
from ..storage.video_storage import VideoStorage
from ..storage.metadata_manager import MetadataManager
from .session_summary import SessionSummaryDialog
from .shot_detail_window import ShotDetailWindow


class VideoThread(QThread):
    """Thread for video processing."""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_manager=None, video_file_manager=None, pose_detector=None, shot_detector=None, video_recorder=None):
        super().__init__()
        self.camera_manager = camera_manager
        self.video_file_manager = video_file_manager
        self.pose_detector = pose_detector
        self.shot_detector = shot_detector
        self.video_recorder = video_recorder
        self.running = False
        self.use_video_file = video_file_manager is not None
        
    def run(self):
        self.running = True
        import time
        
        frame_count = 0
        
        while self.running:
            # Get frame from appropriate source
            if self.use_video_file and self.video_file_manager:
                result = self.video_file_manager.read_frame()
                if result is None:
                    time.sleep(0.033)  # Sleep if paused or end of video
                    continue
                ret, frame = result
                # Add delay for video playback speed
                if self.video_file_manager.frame_delay_ms > 0:
                    time.sleep(self.video_file_manager.frame_delay_ms / 1000.0)
            elif self.camera_manager:
                result = self.camera_manager.read_frame()
                if result is None:
                    continue
                ret, frame = result
            else:
                time.sleep(0.033)
                continue
                
            if not ret:
                continue
            
            # Validate frame
            if frame is None or frame.size == 0:
                print(f"[VideoThread] WARNING: Invalid frame received, skipping")
                sys.stdout.flush()
                continue
            
            # Ensure frame has valid dimensions
            if len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
                print(f"[VideoThread] WARNING: Frame has invalid dimensions: {frame.shape if frame is not None else 'None'}")
                sys.stdout.flush()
                continue
            
            frame_count += 1
            
            try:
                # Log every 30 frames (roughly once per second at 30fps)
                if frame_count % 30 == 0:
                    print(f"[VideoThread] Processing frame {frame_count}, mode: {'video_file' if self.use_video_file else 'camera'}")
                    sys.stdout.flush()
                    
                # Add to video recorder buffer (use original frame)
                self.video_recorder.add_frame(frame)
                
                # Process pose detection first (but don't draw yet)
                # Note: process_frame doesn't modify the original frame
                pose_data = None
                landmarks = None
                if self.pose_detector:
                    try:
                        # Make a copy to avoid any potential memory issues
                        frame_copy = frame.copy()
                        pose_data = self.pose_detector.process_frame(frame_copy)
                        if pose_data:
                            landmarks = pose_data.get('landmarks')
                    except Exception as e:
                        print(f"[VideoThread] ERROR in pose detection: {e}")
                        sys.stdout.flush()
                
                # Process shot detection using player motion dynamics
                if self.shot_detector:
                    try:
                        self.shot_detector.process_frame(frame, landmarks)
                    except Exception as e:
                        print(f"[VideoThread] ERROR in shot detection: {e}")
                        sys.stdout.flush()
                
                # Draw pose overlay on frame
                if self.pose_detector and pose_data:
                    try:
                        frame = self.pose_detector.draw_pose(frame, pose_data)
                    except Exception as e:
                        print(f"[VideoThread] ERROR in pose drawing: {e}")
                        sys.stdout.flush()
                
                # Draw shot state
                if self.shot_detector and self.shot_detector.is_shooting:
                    try:
                        cv2.putText(frame, "SHOOTING", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    except Exception as e:
                        print(f"[VideoThread] ERROR drawing shot state: {e}")
                        sys.stdout.flush()
                
                # Add frame during recording
                if self.video_recorder.is_recording:
                    self.video_recorder.add_recording_frame(frame)
                
                # Validate frame before emitting signal
                # Make a copy to ensure thread safety and prevent segfaults
                if frame is not None and frame.size > 0 and len(frame.shape) >= 2:
                    try:
                        # Create a deep copy using tobytes/frombuffer to avoid memory issues
                        # This ensures the data is completely independent
                        frame_copy = frame.copy()
                        frame_copy = np.ascontiguousarray(frame_copy)
                        
                        # Only emit if frame is valid
                        if frame_copy.size > 0 and frame_copy.shape[0] > 0 and frame_copy.shape[1] > 0:
                            self.frame_ready.emit(frame_copy)
                    except Exception as e:
                        print(f"[VideoThread] ERROR emitting frame signal: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                else:
                    print(f"[VideoThread] WARNING: Skipping invalid frame emission")
                    sys.stdout.flush()
                
            except Exception as e:
                print(f"[VideoThread] CRITICAL ERROR processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                continue
            
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Basketball Trainer - Klay Thompson Coach")
        self.setGeometry(100, 100, 1280, 720)
        
        # Initialize components
        self.camera_manager = CameraManager()
        self.video_file_manager = VideoFileManager()
        self.video_recorder = VideoRecorder()
        self.current_mode = "camera"  # "camera" or "video_file"
        
        # Initialize pose detector (may take a moment to load models)
        # Delay initialization to avoid OpenGL issues during startup
        self.pose_detector = None
        try:
            print("[MainWindow] Initializing pose detector...")
            sys.stdout.flush()
            # Use a timer to delay initialization slightly to let GUI settle
            QTimer.singleShot(100, self._initialize_pose_detector)
        except Exception as e:
            print(f"[MainWindow] ERROR initializing pose detector: {e}")
            sys.stdout.flush()
            QMessageBox.warning(self, "Pose Detector Warning", f"Failed to initialize pose detector: {str(e)}\n\nApp will continue but pose detection may not work.")
            self.pose_detector = None
        
        # Initialize AI components
        try:
            self.gemini_client = GeminiClient()
            self.critique_generator = CritiqueGenerator(self.gemini_client)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize Gemini: {str(e)}")
            sys.exit(1)
        
        # Initialize storage
        self.video_storage = VideoStorage()
        self.metadata_manager = MetadataManager()
        
        # Initialize session management
        self.session_manager = SessionManager()
        self.shot_analyzer = ShotAnalyzer(
            self.critique_generator,
            self.video_storage,
            self.metadata_manager
        )
        
        # Shot detection
        self.shot_detector = ShotDetector(shot_callback=self.on_shot_detected)
        
        # Video thread
        self.video_thread = None
        
        # UI state
        self.current_shot_number = 0
        self.show_mesh = True
        self.shot_data = {}  # {shot_number: {'video_path': str, 'critique': str, 'shot_made': bool}}
        
        self.init_ui()
        self.setup_camera()
    
    def _initialize_pose_detector(self):
        """Initialize pose detector after GUI is ready."""
        try:
            print("[MainWindow] Creating pose detector...")
            sys.stdout.flush()
            self.pose_detector = PoseDetector(show_mesh=True)
            print("[MainWindow] Pose detector initialized successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"[MainWindow] ERROR creating pose detector: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            QMessageBox.warning(self, "Pose Detector Warning", f"Failed to initialize pose detector: {str(e)}\n\nApp will continue but pose detection may not work.")
            self.pose_detector = None
        
    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Camera", "camera")
        self.mode_combo.addItem("Video File", "video_file")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Camera selection (shown in camera mode)
        control_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        control_layout.addWidget(self.camera_combo)
        
        # Add a connect button
        self.connect_camera_btn = QPushButton("Connect Camera")
        self.connect_camera_btn.clicked.connect(self.on_camera_changed)
        control_layout.addWidget(self.connect_camera_btn)
        
        # Video file controls (shown in video file mode)
        self.load_video_btn = QPushButton("Load Video File")
        self.load_video_btn.clicked.connect(self.load_video_file)
        self.load_video_btn.hide()
        control_layout.addWidget(self.load_video_btn)
        
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.hide()
        self.play_pause_btn.setEnabled(False)
        control_layout.addWidget(self.play_pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.hide()
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        control_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)  # 0.1x
        self.speed_slider.setMaximum(500)  # 5.0x
        self.speed_slider.setValue(100)  # 1.0x
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(50)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.speed_slider.hide()
        control_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.hide()
        control_layout.addWidget(self.speed_label)
        
        # Mesh toggle
        self.mesh_checkbox = QCheckBox("Show Pose Mesh")
        self.mesh_checkbox.setChecked(True)
        self.mesh_checkbox.stateChanged.connect(self.on_mesh_toggled)
        control_layout.addWidget(self.mesh_checkbox)
        
        control_layout.addStretch()
        
        # Session controls
        self.start_session_btn = QPushButton("Start Session")
        self.start_session_btn.clicked.connect(self.start_session)
        control_layout.addWidget(self.start_session_btn)
        
        self.end_session_btn = QPushButton("End Session")
        self.end_session_btn.clicked.connect(self.end_session)
        self.end_session_btn.setEnabled(False)
        control_layout.addWidget(self.end_session_btn)
        
        layout.addLayout(control_layout)
        
        # Video progress slider (for video file mode)
        self.video_progress_layout = QHBoxLayout()
        self.video_progress_label = QLabel("00:00 / 00:00")
        self.video_progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_progress_slider.setEnabled(False)
        self.video_progress_slider.sliderPressed.connect(self.on_progress_pressed)
        self.video_progress_slider.sliderReleased.connect(self.on_progress_released)
        self.video_progress_slider.valueChanged.connect(self.on_progress_changed)
        self.video_progress_layout.addWidget(self.video_progress_label)
        self.video_progress_layout.addWidget(self.video_progress_slider)
        self.video_progress_widget = QWidget()
        self.video_progress_widget.setLayout(self.video_progress_layout)
        self.video_progress_widget.hide()
        layout.addWidget(self.video_progress_widget)
        
        # Timer for updating video progress
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_video_progress)
        self.progress_timer.setInterval(100)  # Update every 100ms
        self.progress_slider_dragging = False
        
        # Create splitter for left sidebar and main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left sidebar - Shot summaries
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setMaximumWidth(300)
        sidebar_widget.setMinimumWidth(250)
        
        sidebar_label = QLabel("Shot Feedback")
        sidebar_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        sidebar_layout.addWidget(sidebar_label)
        
        self.shot_list = QListWidget()
        self.shot_list.setStyleSheet("""
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)
        self.shot_list.itemDoubleClicked.connect(self.on_shot_item_double_clicked)
        sidebar_layout.addWidget(self.shot_list)
        
        splitter.addWidget(sidebar_widget)
        
        # Right side - Video display
        video_widget = QWidget()
        video_layout = QVBoxLayout()
        video_widget.setLayout(video_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1280, 720)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        
        splitter.addWidget(video_widget)
        
        # Set splitter sizes (sidebar: 250px, video: rest)
        splitter.setSizes([250, 1000])
        
        layout.addWidget(splitter)
        
    def setup_camera(self):
        """Setup camera and populate camera list."""
        try:
            available_cameras = self.camera_manager.get_available_cameras()
            self.camera_combo.clear()
            
            for cam_idx in available_cameras:
                # Try to get a better name for the camera
                camera_name = self.camera_manager.get_camera_name(cam_idx)
                self.camera_combo.addItem(camera_name, cam_idx)
            
            if available_cameras:
                # Don't auto-connect - let user select manually and click "Connect Camera"
                self.camera_combo.setCurrentIndex(0)
                print(f"Found {len(available_cameras)} camera(s): {available_cameras}")
            else:
                self.connect_camera_btn.setEnabled(False)
                QMessageBox.warning(self, "No Camera", "No cameras found! Please check camera permissions in System Settings.")
        except Exception as e:
            self.connect_camera_btn.setEnabled(False)
            QMessageBox.warning(self, "Camera Setup Error", f"Error setting up cameras: {str(e)}\n\nYou can still use the app, but camera features may not work.")
    
    def on_camera_changed(self):
        """Handle camera selection change."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        camera_index = self.camera_combo.currentData()
        if camera_index is not None:
            try:
                if self.camera_manager.connect(camera_index):
                    self.start_video_thread()
                else:
                    QMessageBox.warning(
                        self, 
                        "Camera Error", 
                        f"Failed to connect to camera {camera_index}.\n\n"
                        "This might be a Continuity Camera (iPhone).\n"
                        "Try selecting a different camera or check camera permissions."
                    )
            except Exception as e:
                QMessageBox.warning(self, "Camera Error", f"Error connecting to camera: {str(e)}")
    
    def on_mode_changed(self, index):
        """Handle mode change between camera and video file."""
        mode = self.mode_combo.currentData()
        self.current_mode = mode
        
        # Stop current video thread
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        if mode == "camera":
            # Show camera controls
            self.camera_combo.show()
            self.connect_camera_btn.show()
            self.load_video_btn.hide()
            self.play_pause_btn.hide()
            self.stop_btn.hide()
            self.speed_slider.hide()
            self.speed_label.hide()
            self.video_progress_widget.hide()
            self.progress_timer.stop()
            
            # Disconnect video file if connected
            if self.video_file_manager.is_connected:
                self.video_file_manager.disconnect()
        else:
            # Show video file controls
            self.camera_combo.hide()
            self.connect_camera_btn.hide()
            self.load_video_btn.show()
            self.play_pause_btn.show()
            self.stop_btn.show()
            self.speed_slider.show()
            self.speed_label.show()
            self.video_progress_widget.show()
            
            # Disconnect camera if connected
            if self.camera_manager.is_connected:
                self.camera_manager.disconnect()
    
    def load_video_file(self):
        """Load a video file for playback."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if file_path:
            print(f"[MainWindow] Loading video file: {file_path}")
            sys.stdout.flush()
            if self.video_file_manager.load_video(file_path):
                fps = self.video_file_manager.get_fps()
                frame_count = self.video_file_manager.get_frame_count()
                width, height = self.video_file_manager.get_frame_size() or (0, 0)
                print(f"[MainWindow] Video loaded: {frame_count} frames, {fps} fps, {width}x{height}")
                sys.stdout.flush()
                self.play_pause_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.video_progress_slider.setEnabled(True)
                self.video_progress_slider.setMaximum(frame_count)
                self.start_video_thread()
                self.progress_timer.start()
                print(f"[MainWindow] Video thread started")
                sys.stdout.flush()
                QMessageBox.information(self, "Video Loaded", f"Video loaded successfully!\n\nFile: {file_path}")
            else:
                print(f"[MainWindow] ERROR: Failed to load video file")
                sys.stdout.flush()
                QMessageBox.warning(self, "Error", "Failed to load video file. Please check the file format.")
    
    def toggle_playback(self):
        """Toggle video playback (play/pause)."""
        if not self.video_file_manager.is_connected:
            return
        
        if self.video_file_manager.is_playing:
            self.video_file_manager.pause()
            self.play_pause_btn.setText("Play")
            print(f"[MainWindow] Video paused")
            sys.stdout.flush()
        else:
            self.video_file_manager.play()
            self.play_pause_btn.setText("Pause")
            print(f"[MainWindow] Video playing")
            sys.stdout.flush()
    
    def stop_video(self):
        """Stop video playback and reset to beginning."""
        if self.video_file_manager.is_connected:
            self.video_file_manager.stop()
            self.play_pause_btn.setText("Play")
            self.update_video_progress()
    
    def on_speed_changed(self, value):
        """Handle playback speed change."""
        speed = value / 100.0  # Convert slider value (10-500) to speed (0.1-5.0)
        self.video_file_manager.set_playback_speed(speed)
        self.speed_label.setText(f"{speed:.1f}x")
    
    def update_video_progress(self):
        """Update video progress slider and label."""
        if not self.video_file_manager.is_connected or self.progress_slider_dragging:
            return
        
        current_frame = self.video_file_manager.get_current_frame()
        total_frames = self.video_file_manager.get_frame_count()
        fps = self.video_file_manager.get_fps()
        
        if total_frames > 0:
            self.video_progress_slider.setValue(current_frame)
            
            # Update time label
            current_time = current_frame / fps if fps > 0 else 0
            total_time = total_frames / fps if fps > 0 else 0
            
            current_min = int(current_time // 60)
            current_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.video_progress_label.setText(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
    
    def on_progress_pressed(self):
        """Handle progress slider being pressed (start dragging)."""
        self.progress_slider_dragging = True
    
    def on_progress_released(self):
        """Handle progress slider being released (end dragging)."""
        if self.video_file_manager.is_connected:
            frame_number = self.video_progress_slider.value()
            self.video_file_manager.seek_to_frame(frame_number)
        self.progress_slider_dragging = False
    
    def on_progress_changed(self, value):
        """Handle progress slider value change (while dragging)."""
        # This is called during dragging, but we only seek on release
        pass
    
    def on_mesh_toggled(self, state):
        """Handle pose mesh toggle."""
        self.show_mesh = (state == Qt.CheckState.Checked.value)
        if self.pose_detector:
            self.pose_detector.show_mesh = self.show_mesh
    
    def start_video_thread(self):
        """Start video processing thread."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        
        print(f"[MainWindow] Starting video thread in {self.current_mode} mode")
        sys.stdout.flush()
        if self.current_mode == "video_file":
            self.video_thread = VideoThread(
                camera_manager=None,
                video_file_manager=self.video_file_manager,
                pose_detector=self.pose_detector,
                shot_detector=self.shot_detector,
                video_recorder=self.video_recorder
            )
            print(f"[MainWindow] Video thread created for video_file mode")
            sys.stdout.flush()
        else:
            self.video_thread = VideoThread(
                camera_manager=self.camera_manager,
                video_file_manager=None,
                pose_detector=self.pose_detector,
                shot_detector=self.shot_detector,
                video_recorder=self.video_recorder
            )
            print(f"[MainWindow] Video thread created for camera mode")
            sys.stdout.flush()
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.start()
        print(f"[MainWindow] Video thread started")
        sys.stdout.flush()
    
    def update_video_display(self, frame):
        """Update video display with new frame."""
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return
            
            if len(frame.shape) < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
                return
            
            # Make a copy to avoid memory issues
            frame = frame.copy()
            
            # Draw stats on frame
            stats = self.session_manager.get_stats()
            stats_text = f"Shots: {stats['total_shots']} | Made: {stats['shots_made']} | Missed: {stats['shots_missed']}"
            try:
                cv2.putText(frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception as e:
                print(f"[MainWindow] ERROR drawing stats: {e}")
                sys.stdout.flush()
            
            # Convert to QImage - use a safer method that copies the data
            try:
                # Ensure frame is contiguous and in the right format
                frame = np.ascontiguousarray(frame)
                height, width, channel = frame.shape
                
                if channel != 3:
                    print(f"[MainWindow] WARNING: Frame has {channel} channels, expected 3")
                    sys.stdout.flush()
                    return
                
                # Convert BGR to RGB for QImage
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                
                # Create QImage using bytes() to ensure data is copied
                bytes_per_line = 3 * width
                q_image = QImage(
                    rgb_frame.tobytes(), 
                    width, 
                    height, 
                    bytes_per_line, 
                    QImage.Format.Format_RGB888
                )
                
                if q_image.isNull():
                    print(f"[MainWindow] WARNING: QImage is null")
                    sys.stdout.flush()
                    return
                
                # Scale to fit label
                pixmap = QPixmap.fromImage(q_image)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.video_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.video_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"[MainWindow] ERROR converting frame to QImage: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        except Exception as e:
            print(f"[MainWindow] CRITICAL ERROR in update_video_display: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    def start_session(self):
        """Start a new training session."""
        session_id = self.session_manager.start_session()
        self.current_shot_number = 0
        self.start_session_btn.setEnabled(False)
        self.end_session_btn.setEnabled(True)
        QMessageBox.information(self, "Session Started", f"Session {session_id[:8]}... started!")
    
    def end_session(self):
        """End the current session."""
        if not self.session_manager.is_active:
            return
        
        session_id = self.session_manager.get_session_id()
        
        # End session first
        metadata = self.session_manager.end_session()
        
        if metadata and metadata.get('shots'):
            # Generate session summary
            critiques = [shot.get('critique', '') for shot in metadata['shots'] if shot.get('critique')]
            if critiques:
                summary = self.critique_generator.generate_session_summary(critiques)
                self.metadata_manager.update_session_summary(session_id, summary)
                
                # Reload metadata with summary
                metadata = self.metadata_manager.load_session_metadata(session_id)
            
            # Show summary dialog
            if metadata:
                dialog = SessionSummaryDialog(metadata, self)
                dialog.exec()
        
        self.start_session_btn.setEnabled(True)
        self.end_session_btn.setEnabled(False)
    
    def on_shot_detected(self):
        """Handle shot detection."""
        if not self.session_manager.is_active:
            return
        
        self.current_shot_number += 1
        
        # Start recording (includes buffer)
        self.video_recorder.start_recording()
        
        # Record for 3 seconds after shot detection to capture if shot goes in
        QTimer.singleShot(3000, self.process_shot)
    
    def process_shot(self):
        """Process detected shot."""
        # Stop recording and get frames
        frames = self.video_recorder.stop_recording()
        
        if not frames:
            return
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Analyze shot asynchronously (Gemini will determine if shot was made)
        session_id = self.session_manager.get_session_id()
        self.shot_analyzer.analyze_shot(
            session_id,
            self.current_shot_number,
            frames,
            width,
            height,
            shot_made=None,  # Let Gemini determine
            callback=self.on_shot_analyzed
        )
    
    def on_shot_analyzed(self, shot_number: int, shot_made: bool, critique: str, video_path: str):
        """Handle completed shot analysis.
        
        Args:
            shot_number: Shot number
            shot_made: Whether shot was made
            critique: Klay's critique
            video_path: Path to saved video
        """
        # Reload session metadata to get updated stats
        session_id = self.session_manager.get_session_id()
        if session_id:
            self.session_manager.session_metadata = self.metadata_manager.load_session_metadata(session_id)
        
        # Store shot data
        self.shot_data[shot_number] = {
            'video_path': video_path,
            'critique': critique,
            'shot_made': shot_made
        }
        
        # Add to sidebar list
        result_text = "✓ Made" if shot_made else "✗ Missed"
        preview_text = critique[:80] + "..." if len(critique) > 80 else critique
        item_text = f"Shot #{shot_number} - {result_text}\n{preview_text}"
        
        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, shot_number)  # Store shot number
        self.shot_list.addItem(item)
        
        # Scroll to bottom to show newest
        self.shot_list.scrollToBottom()
        
        print(f"Shot {shot_number} analyzed: {critique[:100]}...")
    
    def on_shot_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on shot item to open detail window.
        
        Args:
            item: The list item that was double-clicked
        """
        shot_number = item.data(Qt.ItemDataRole.UserRole)
        if shot_number and shot_number in self.shot_data:
            data = self.shot_data[shot_number]
            detail_window = ShotDetailWindow(
                data['video_path'],
                data['critique'],
                data['shot_made'],
                shot_number,
                self
            )
            detail_window.exec()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        self.camera_manager.disconnect()
        event.accept()

