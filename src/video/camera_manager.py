"""Camera management for video capture."""
import cv2
import os
import sys
from contextlib import contextmanager
from typing import Optional, List


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class CameraManager:
    """Manages camera capture and device selection."""
    
    def __init__(self, camera_index: int = 0):
        """Initialize camera manager.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        
    def get_available_cameras(self) -> List[int]:
        """Enumerate available cameras.
        
        Returns:
            List of available camera indices
        """
        available = []
        consecutive_failures = 0
        max_consecutive_failures = 5  # Stop after 5 consecutive failures
        max_cameras_to_check = 20  # Check up to 20 camera indices
        
        # Suppress OpenCV error messages during enumeration
        for i in range(max_cameras_to_check):
            camera_works = False
            with suppress_stderr():
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        # Try to read a frame to verify it's actually working
                        ret, _ = cap.read()
                        if ret:
                            available.append(i)
                            camera_works = True
                            consecutive_failures = 0
                        cap.release()
                    else:
                        # Make sure to release even if not opened
                        try:
                            cap.release()
                        except:
                            pass
                except Exception:
                    # Camera access failed
                    pass
            
            if not camera_works:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            # Stop early if we hit consecutive failures (likely out of bounds)
            # But only stop if we've found at least one camera and had many failures
            # This allows finding cameras at non-consecutive indices
            if consecutive_failures >= max_consecutive_failures and len(available) > 0:
                break
        
        return available
    
    def get_camera_name(self, camera_index: int) -> str:
        """Get a descriptive name for a camera.
        
        Args:
            camera_index: Index of the camera
            
        Returns:
            Camera name or "Camera {index}" if name unavailable
        """
        with suppress_stderr():
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    # Try to get backend name (might give us some info)
                    backend = cap.getBackendName()
                    # Try to get camera name if available (OpenCV 4.5+)
                    try:
                        # Some backends support getting device name
                        name = f"Camera {camera_index}"
                        # On macOS, we could try to get more info, but OpenCV is limited
                        cap.release()
                        return name
                    except:
                        cap.release()
                        return f"Camera {camera_index}"
                cap.release()
            except:
                pass
        return f"Camera {camera_index}"
    
    def connect(self, camera_index: Optional[int] = None) -> bool:
        """Connect to a camera.
        
        Args:
            camera_index: Camera index to connect to (uses self.camera_index if None)
            
        Returns:
            True if connection successful, False otherwise
        """
        if camera_index is not None:
            self.camera_index = camera_index
            
        if self.cap is not None:
            self.disconnect()
        
        # Suppress OpenCV error messages during connection attempt
        with suppress_stderr():
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            self.is_connected = False
            return False
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_connected = True
        return True
    
    def read_frame(self) -> Optional[tuple]:
        """Read a frame from the camera.
        
        Returns:
            (success, frame) tuple or None if not connected
        """
        if not self.is_connected or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        
        if not ret:
            # Check if camera is still connected
            if not self.cap.isOpened():
                self.is_connected = False
            return None
            
        return (ret, frame)
    
    def disconnect(self):
        """Disconnect from the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
    
    def get_frame_size(self) -> Optional[tuple]:
        """Get the current frame size.
        
        Returns:
            (width, height) tuple or None if not connected
        """
        if not self.is_connected or self.cap is None:
            return None
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()

