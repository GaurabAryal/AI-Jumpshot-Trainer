"""Pose detection using MediaPipe."""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Tuple


class PoseDetector:
    """Detects and tracks human pose using MediaPipe."""
    
    def __init__(self, show_mesh: bool = True):
        """Initialize pose detector.
        
        Args:
            show_mesh: Whether to show body mesh overlay
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.show_mesh = show_mesh
        self.last_landmarks = None
        
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process frame for pose detection.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with landmarks and drawing info, or None
        """
        if frame is None:
            return None
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process pose
        results = self.pose.process(rgb_frame)
        
        # Convert back to BGR for drawing
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            self.last_landmarks = results.pose_landmarks
            
            # Extract landmark positions
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            return {
                'landmarks': landmarks,
                'pose_landmarks': results.pose_landmarks,
                'frame': frame
            }
        
        return None
    
    def draw_pose(self, frame: np.ndarray, pose_data: Optional[Dict]) -> np.ndarray:
        """Draw pose mesh and skeleton on frame.
        
        Args:
            frame: Frame to draw on
            pose_data: Pose data from process_frame
            
        Returns:
            Frame with pose overlay
        """
        if pose_data is None or pose_data.get('pose_landmarks') is None:
            return frame
            
        if self.show_mesh:
            # Draw pose landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame,
                pose_data['pose_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame
    
    def get_key_landmarks(self, landmarks: Dict) -> Optional[Dict]:
        """Extract key landmarks for shooting form analysis.
        
        Args:
            landmarks: Full landmarks dictionary
            
        Returns:
            Dictionary with key shooting form landmarks
        """
        if not landmarks:
            return None
            
        # MediaPipe pose landmark indices
        key_points = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'nose': 0
        }
        
        key_landmarks = {}
        for name, idx in key_points.items():
            if idx in landmarks:
                key_landmarks[name] = landmarks[idx]
                
        return key_landmarks
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'pose'):
            self.pose.close()

