"""Video file storage and organization."""
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class VideoStorage:
    """Manages video file storage."""
    
    def __init__(self, base_path: str = "data/videos"):
        """Initialize video storage.
        
        Args:
            base_path: Base directory for video storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def create_session_directory(self, session_id: str) -> Path:
        """Create directory for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Path to session directory
        """
        session_dir = self.base_path / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def get_shot_video_path(self, session_id: str, shot_number: int, 
                           shot_made: bool, with_critique: bool = False) -> Path:
        """Get path for a shot video.
        
        Args:
            session_id: Session identifier
            shot_number: Shot number in session
            shot_made: Whether shot was made
            with_critique: Whether this is the final video with critique overlay
            
        Returns:
            Path to video file
        """
        session_dir = self.create_session_directory(session_id)
        result = "made" if shot_made else "missed"
        suffix = "_with_critique" if with_critique else ""
        filename = f"shot_{shot_number:03d}_{result}{suffix}.mp4"
        return session_dir / filename
    
    def save_video(self, video_path: Path, frames: list, width: int, height: int, fps: int = 30) -> bool:
        """Save frames as video file.
        
        Args:
            video_path: Path to save video
            frames: List of frames to save
            width: Frame width
            height: Frame height
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        if not frames:
            return False
            
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            return False
            
        for frame in frames:
            out.write(frame)
            
        out.release()
        return True

