"""Video text overlay using FFmpeg."""
import subprocess
import os
import platform
from pathlib import Path
from typing import Optional


class VideoOverlay:
    """Adds text overlays to videos using FFmpeg."""
    
    def __init__(self):
        """Initialize video overlay."""
        pass
    
    def add_text_overlay(self, input_path: str, output_path: str, 
                        text: str, position: str = "bottom") -> bool:
        """Add text overlay to video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            text: Text to overlay
            position: Text position ("top", "center", "bottom")
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            return False
            
        # Determine position coordinates
        if position == "top":
            y_pos = "50"
        elif position == "center":
            y_pos = "(h-text_h)/2"
        else:  # bottom
            y_pos = "h-th-50"
        
        # Create FFmpeg filter for text overlay
        # Split text into multiple lines if too long
        lines = self._split_text_into_lines(text, max_chars=60)
        filter_parts = []
        
        # Try to find a system font
        font_path = self._find_system_font()
        
        for i, line in enumerate(lines):
            # Escape special characters for FFmpeg
            escaped_line = line.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")
            y_offset = int(y_pos) + (i * 40) if isinstance(y_pos, str) and y_pos.isdigit() else f"{y_pos}+{i*40}"
            font_param = f"fontfile={font_path}:" if font_path else ""
            filter_parts.append(
                f"drawtext=text='{escaped_line}':{font_param}"
                f"fontsize=24:fontcolor=white:x=(w-text_w)/2:y={y_offset}:"
                f"box=1:boxcolor=black@0.5:boxborderw=5"
            )
        
        filter_complex = ",".join(filter_parts)
        
        # FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', filter_complex,
            '-codec:a', 'copy',
            '-y',  # Overwrite output file
            output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False
        except FileNotFoundError:
            print("FFmpeg not found. Please install FFmpeg.")
            return False
    
    def _split_text_into_lines(self, text: str, max_chars: int = 60) -> list:
        """Split text into lines for display.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per line
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_chars and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_length
                
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines if lines else [text]
    
    def _find_system_font(self) -> Optional[str]:
        """Find a system font for FFmpeg.
        
        Returns:
            Path to font file or None
        """
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            font_paths = [
                '/System/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/Arial.ttf',
                '/Library/Fonts/Arial.ttf'
            ]
        elif system == 'Windows':
            font_paths = [
                'C:/Windows/Fonts/arial.ttf',
                'C:/Windows/Fonts/calibri.ttf'
            ]
        else:  # Linux
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
                
        return None

