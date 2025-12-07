"""Orchestrates shot analysis workflow."""
import threading
import shutil
from pathlib import Path
from typing import Optional, Callable
from ..ai.critique_generator import CritiqueGenerator
from ..storage.video_storage import VideoStorage
from ..storage.metadata_manager import MetadataManager
from ..utils.video_overlay import VideoOverlay


class ShotAnalyzer:
    """Orchestrates the complete shot analysis workflow."""
    
    def __init__(self, critique_generator: CritiqueGenerator, 
                 video_storage: VideoStorage,
                 metadata_manager: MetadataManager):
        """Initialize shot analyzer.
        
        Args:
            critique_generator: Critique generator instance
            video_storage: Video storage instance
            metadata_manager: Metadata manager instance
        """
        self.critique_generator = critique_generator
        self.video_storage = video_storage
        self.metadata_manager = metadata_manager
        self.video_overlay = VideoOverlay()
        self.analysis_callback: Optional[Callable] = None
        
    def analyze_shot(self, session_id: str, shot_number: int, 
                    frames: list, width: int, height: int, 
                    shot_made: Optional[bool] = None, callback: Optional[Callable] = None):
        """Analyze a shot asynchronously.
        
        Args:
            session_id: Session identifier
            shot_number: Shot number
            frames: Video frames
            width: Frame width
            height: Frame height
            shot_made: Whether shot was made (if None, will be determined by Gemini)
            callback: Optional callback when analysis completes
        """
        self.analysis_callback = callback
        
        # Run analysis in background thread
        thread = threading.Thread(
            target=self._analyze_shot_sync,
            args=(session_id, shot_number, frames, width, height, shot_made)
        )
        thread.daemon = True
        thread.start()
    
    def _analyze_shot_sync(self, session_id: str, shot_number: int,
                          frames: list, width: int, height: int, shot_made: Optional[bool]):
        """Synchronous shot analysis workflow.
        
        Args:
            session_id: Session identifier
            shot_number: Shot number
            frames: Video frames
            width: Frame width
            height: Frame height
            shot_made: Whether shot was made
        """
        try:
            # 1. Save initial video clip (use temporary path first since we don't know result yet)
            # We'll use a placeholder for shot_made, then rename after determining result
            temp_shot_made = shot_made if shot_made is not None else False
            initial_video_path = self.video_storage.get_shot_video_path(
                session_id, shot_number, temp_shot_made, with_critique=False
            )
            
            if not self.video_storage.save_video(initial_video_path, frames, width, height):
                print(f"Failed to save initial video for shot {shot_number}")
                return
            
            # 2. Determine if shot was made using Gemini (if not already determined)
            if shot_made is None:
                print(f"[ShotAnalyzer] Determining shot result for shot {shot_number} using Gemini...")
                shot_made = self.critique_generator.determine_shot_result(str(initial_video_path))
                print(f"[ShotAnalyzer] Shot {shot_number} result: {'MADE' if shot_made else 'MISSED'}")
                
                # If result changed, we need to save to correct path
                if shot_made != temp_shot_made:
                    correct_video_path = self.video_storage.get_shot_video_path(
                        session_id, shot_number, shot_made, with_critique=False
                    )
                    # Move/rename file to correct path
                    Path(correct_video_path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(initial_video_path), str(correct_video_path))
                    initial_video_path = correct_video_path
            
            # 3. Generate critique using Gemini
            critique = self.critique_generator.generate_shot_critique(
                str(initial_video_path),
                shot_made
            )
            
            # 4. Add critique overlay to video
            final_video_path = self.video_storage.get_shot_video_path(
                session_id, shot_number, shot_made, with_critique=True
            )
            
            if not self.video_overlay.add_text_overlay(
                str(initial_video_path),
                str(final_video_path),
                critique,
                position="bottom"
            ):
                # If overlay fails, use original video
                final_video_path = initial_video_path
            
            # 5. Save shot metadata
            self.metadata_manager.add_shot_metadata(
                session_id,
                shot_number,
                shot_made,
                str(final_video_path),
                critique
            )
            
            # 6. Call callback if provided
            if self.analysis_callback:
                self.analysis_callback(shot_number, shot_made, critique, str(final_video_path))
                
        except Exception as e:
            print(f"Error analyzing shot {shot_number}: {str(e)}")
            if self.analysis_callback:
                self.analysis_callback(shot_number, shot_made, f"Error: {str(e)}", None)

