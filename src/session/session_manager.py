"""Session lifecycle and data management."""
import uuid
from datetime import datetime
from typing import Optional, Callable
from ..storage.metadata_manager import MetadataManager
from ..storage.video_storage import VideoStorage


class SessionManager:
    """Manages training session lifecycle."""
    
    def __init__(self):
        """Initialize session manager."""
        self.metadata_manager = MetadataManager()
        self.video_storage = VideoStorage()
        self.current_session_id: Optional[str] = None
        self.session_metadata: Optional[dict] = None
        self.is_active = False
        
    def start_session(self) -> str:
        """Start a new training session.
        
        Returns:
            Session ID
        """
        if self.is_active:
            self.end_session()
            
        self.current_session_id = str(uuid.uuid4())
        self.session_metadata = self.metadata_manager.create_session_metadata(self.current_session_id)
        self.is_active = True
        return self.current_session_id
    
    def end_session(self) -> Optional[dict]:
        """End the current session.
        
        Returns:
            Final session metadata or None if no active session
        """
        if not self.is_active or self.current_session_id is None:
            return None
            
        if self.session_metadata:
            self.session_metadata['end_time'] = datetime.now().isoformat()
            self.metadata_manager.save_session_metadata(
                self.current_session_id, 
                self.session_metadata
            )
            
        self.is_active = False
        return self.session_metadata
    
    def get_session_id(self) -> Optional[str]:
        """Get current session ID.
        
        Returns:
            Session ID or None
        """
        return self.current_session_id
    
    def get_stats(self) -> dict:
        """Get current session statistics.
        
        Returns:
            Dictionary with shot counts
        """
        if not self.session_metadata:
            return {'total_shots': 0, 'shots_made': 0, 'shots_missed': 0}
            
        return {
            'total_shots': self.session_metadata.get('total_shots', 0),
            'shots_made': self.session_metadata.get('shots_made', 0),
            'shots_missed': self.session_metadata.get('shots_missed', 0)
        }
    
    def add_shot(self, shot_number: int, shot_made: bool, video_path: str, critique: str):
        """Add a shot to the current session.
        
        Args:
            shot_number: Shot number in session
            shot_made: Whether shot was made
            video_path: Path to shot video
            critique: Klay's critique
        """
        if not self.is_active or self.current_session_id is None:
            return
            
        self.metadata_manager.add_shot_metadata(
            self.current_session_id,
            shot_number,
            shot_made,
            video_path,
            critique
        )
        
        # Update local metadata
        if self.session_metadata:
            self.session_metadata = self.metadata_manager.load_session_metadata(
                self.current_session_id
            )

