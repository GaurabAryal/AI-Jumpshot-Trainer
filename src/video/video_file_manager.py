"""Video file management for playback."""
import cv2
import os
from typing import Optional, Tuple
from pathlib import Path


class VideoFileManager:
    """Manages video file playback."""
    
    def __init__(self):
        """Initialize video file manager."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.is_connected = False
        self.is_playing = False
        self.playback_speed = 1.0  # 1.0 = normal speed, 2.0 = 2x speed, etc.
        self.frame_delay_ms = 33  # ~30 fps default (1000ms / 30fps)
        
    def load_video(self, video_path: str) -> bool:
        """Load a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video loaded successfully, False otherwise
        """
        if not os.path.exists(video_path):
            print(f"[VideoFileManager] ERROR: Video file not found: {video_path}")
            return False
        
        if self.cap is not None:
            self.disconnect()
        
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"[VideoFileManager] ERROR: Failed to open video file: {video_path}")
            self.is_connected = False
            return False
        
        self.video_path = video_path
        self.is_connected = True
        self.is_playing = False
        
        # Calculate frame delay based on video FPS and playback speed
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps > 0:
            self.frame_delay_ms = int(1000 / (fps * self.playback_speed))
        else:
            self.frame_delay_ms = 33  # Default to ~30fps
        
        print(f"[VideoFileManager] Video loaded: {frame_count} frames, {fps} fps, {width}x{height}")
        import sys
        sys.stdout.flush()
        return True
    
    def read_frame(self) -> Optional[Tuple[bool, any]]:
        """Read a frame from the video file.
        
        Returns:
            (success, frame) tuple or None if not connected
        """
        if not self.is_connected or self.cap is None:
            return None
        
        if not self.is_playing:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video - loop back to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing = False
            return None
        
        return (ret, frame)
    
    def get_frame_count(self) -> int:
        """Get total number of frames in the video.
        
        Returns:
            Total frame count, or 0 if not connected
        """
        if not self.is_connected or self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_current_frame(self) -> int:
        """Get current frame position.
        
        Returns:
            Current frame number, or 0 if not connected
        """
        if not self.is_connected or self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def get_fps(self) -> float:
        """Get video FPS.
        
        Returns:
            FPS value, or 0 if not connected
        """
        if not self.is_connected or self.cap is None:
            return 0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get the video frame size.
        
        Returns:
            (width, height) tuple or None if not connected
        """
        if not self.is_connected or self.cap is None:
            return None
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier.
        
        Args:
            speed: Playback speed (1.0 = normal, 2.0 = 2x, 0.5 = 0.5x)
        """
        self.playback_speed = max(0.1, min(5.0, speed))  # Clamp between 0.1x and 5x
        if self.is_connected and self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.frame_delay_ms = int(1000 / (fps * self.playback_speed))
    
    def play(self):
        """Start playing the video."""
        if self.is_connected:
            self.is_playing = True
            print(f"[VideoFileManager] Video playback started")
            import sys
            sys.stdout.flush()
    
    def pause(self):
        """Pause the video."""
        self.is_playing = False
    
    def stop(self):
        """Stop and reset the video to the beginning."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.is_playing = False
    
    def seek_to_frame(self, frame_number: int):
        """Seek to a specific frame.
        
        Args:
            frame_number: Frame number to seek to
        """
        if self.is_connected and self.cap is not None:
            total_frames = self.get_frame_count()
            frame_number = max(0, min(frame_number, total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def disconnect(self):
        """Disconnect from the video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.is_playing = False
        self.video_path = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()

