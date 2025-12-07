"""Video recording with buffering for shot capture."""
import cv2
import numpy as np
from collections import deque
from typing import Optional, Callable
import threading
import time


class VideoRecorder:
    """Records video with pre-shot buffering."""
    
    def __init__(self, buffer_seconds: float = 3.0, fps: int = 30):
        """Initialize video recorder.
        
        Args:
            buffer_seconds: Number of seconds to buffer before shot detection
            fps: Frames per second for recording
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self.frame_buffer: deque = deque(maxlen=self.buffer_size)
        self.is_recording = False
        self.current_video_frames: list = []
        self.recording_lock = threading.Lock()
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer.
        
        Args:
            frame: Video frame to add
        """
        if frame is not None:
            self.frame_buffer.append(frame.copy())
    
    def start_recording(self):
        """Start recording a shot (includes buffer)."""
        with self.recording_lock:
            self.is_recording = True
            # Copy buffer frames to current recording
            self.current_video_frames = list(self.frame_buffer)
    
    def add_recording_frame(self, frame: np.ndarray):
        """Add a frame during active recording.
        
        Args:
            frame: Video frame to add
        """
        if self.is_recording and frame is not None:
            self.current_video_frames.append(frame.copy())
    
    def stop_recording(self) -> list:
        """Stop recording and return all frames.
        
        Returns:
            List of frames from buffer start to recording end
        """
        with self.recording_lock:
            self.is_recording = False
            frames = self.current_video_frames.copy()
            self.current_video_frames.clear()
            return frames
    
    def save_video(self, frames: list, output_path: str, width: int, height: int) -> bool:
        """Save frames to a video file.
        
        Args:
            frames: List of frames to save
            output_path: Path to save video file
            width: Frame width
            height: Frame height
            
        Returns:
            True if successful, False otherwise
        """
        if not frames:
            return False
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        if not out.isOpened():
            return False
            
        for frame in frames:
            out.write(frame)
            
        out.release()
        return True

