"""Session and shot metadata management."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class MetadataManager:
    """Manages session and shot metadata."""
    
    def __init__(self, base_path: str = "data/sessions"):
        """Initialize metadata manager.
        
        Args:
            base_path: Base directory for metadata storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def create_session_metadata(self, session_id: str) -> Dict:
        """Create new session metadata.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session metadata dictionary
        """
        metadata = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_shots': 0,
            'shots_made': 0,
            'shots_missed': 0,
            'shots': [],
            'summary': None
        }
        return metadata
    
    def save_session_metadata(self, session_id: str, metadata: Dict):
        """Save session metadata to file.
        
        Args:
            session_id: Session identifier
            metadata: Metadata dictionary
        """
        file_path = self.base_path / f"session_{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Load session metadata from file.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        file_path = self.base_path / f"session_{session_id}.json"
        if not file_path.exists():
            return None
            
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def add_shot_metadata(self, session_id: str, shot_number: int, 
                         shot_made: bool, video_path: str, critique: str):
        """Add shot metadata to session.
        
        Args:
            session_id: Session identifier
            shot_number: Shot number
            shot_made: Whether shot was made
            video_path: Path to shot video
            critique: Klay's critique
        """
        metadata = self.load_session_metadata(session_id)
        if metadata is None:
            metadata = self.create_session_metadata(session_id)
            
        shot_data = {
            'shot_number': shot_number,
            'timestamp': datetime.now().isoformat(),
            'shot_made': shot_made,
            'video_path': str(video_path),
            'critique': critique
        }
        
        metadata['shots'].append(shot_data)
        metadata['total_shots'] = len(metadata['shots'])
        metadata['shots_made'] = sum(1 for s in metadata['shots'] if s['shot_made'])
        metadata['shots_missed'] = metadata['total_shots'] - metadata['shots_made']
        
        self.save_session_metadata(session_id, metadata)
    
    def update_session_summary(self, session_id: str, summary: str):
        """Update session with summary.
        
        Args:
            session_id: Session identifier
            summary: Session summary text
        """
        metadata = self.load_session_metadata(session_id)
        if metadata is None:
            return
            
        metadata['summary'] = summary
        metadata['end_time'] = datetime.now().isoformat()
        self.save_session_metadata(session_id, metadata)

