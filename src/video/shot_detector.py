"""Automatic shot detection using player motion dynamics."""
import cv2
import numpy as np
from typing import Optional, Callable, Tuple
from collections import deque


class ShotDetector:
    """Detects basketball shots automatically using pose-based motion analysis."""
    
    # MediaPipe pose landmark indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0
    
    def __init__(self, shot_callback: Optional[Callable] = None):
        """Initialize shot detector.
        
        Args:
            shot_callback: Callback function called when shot is detected
        """
        self.shot_callback = shot_callback
        
        # Shot detection parameters (tuned for pose-based detection)
        self.wrist_upward_velocity_threshold = -0.012  # Negative = upward movement (more strict)
        self.min_wrist_height = 0.25  # Minimum wrist height (fraction of frame) to consider shot
        self.max_wrist_height = 0.85  # Maximum wrist height for valid shot
        self.elbow_extension_threshold = 0.08  # Minimum increase in elbow-wrist distance
        self.min_confidence = 0.5  # Minimum visibility/confidence for landmarks
        self.min_wrist_above_shoulder = 0.05  # Wrist must be at least this much above shoulder (normalized)
        
        # State tracking
        self.wrist_positions: deque = deque(maxlen=20)  # Track last 20 frames (~0.67s at 30fps)
        self.elbow_positions: deque = deque(maxlen=20)
        self.shoulder_positions: deque = deque(maxlen=20)
        self.arm_extension_history: deque = deque(maxlen=20)  # Track elbow-wrist distances
        
        self.is_shooting = False
        self.shot_detected = False
        self.frames_since_release = 0
        self.max_frames_after_release = 90  # 3 seconds at 30fps
        self.shooting_arm = None  # 'left' or 'right'
        
    def process_frame(self, frame: np.ndarray, player_landmarks: Optional[dict] = None):
        """Process frame for shot detection using player motion.
        
        Args:
            frame: Current video frame
            player_landmarks: MediaPipe pose landmarks dictionary
        """
        if frame is None:
            return
            
        if player_landmarks is None:
            # Reset if no pose detected for too long
            if self.frames_since_release > self.max_frames_after_release:
                self.reset_state()
            else:
                self.frames_since_release += 1
            return
        
        height, width = frame.shape[:2]
        
        # Extract key landmarks for shooting arm detection
        # Try right arm first (most common shooting arm), then left
        wrist_pos, elbow_pos, shoulder_pos, arm_side = self._get_shooting_arm_landmarks(
            player_landmarks, height, width
        )
        
        if wrist_pos is None or elbow_pos is None or shoulder_pos is None:
            # Not enough landmarks visible
            if self.frames_since_release > self.max_frames_after_release:
                self.reset_state()
            else:
                self.frames_since_release += 1
            return
        
        # Store positions for velocity calculation
        self.wrist_positions.append(wrist_pos)
        self.elbow_positions.append(elbow_pos)
        self.shoulder_positions.append(shoulder_pos)
        
        # Calculate arm extension (distance between elbow and wrist)
        arm_extension = np.sqrt(
            (wrist_pos[0] - elbow_pos[0])**2 + 
            (wrist_pos[1] - elbow_pos[1])**2
        )
        self.arm_extension_history.append(arm_extension)
        
        # Normalized positions for height checks
        normalized_wrist_y = wrist_pos[1] / height
        normalized_shoulder_y = shoulder_pos[1] / height
        
        # Need at least 7 frames of history for reliable detection
        if len(self.wrist_positions) >= 7:
            # Calculate wrist velocity
            wrist_velocity = self._calculate_velocity(self.wrist_positions)
            
            if wrist_velocity is None:
                return
            
            vertical_velocity = wrist_velocity[1]  # Negative = upward
            
            # CRITICAL CHECK: Wrist must be above shoulder (prevents false positives when bending down)
            wrist_above_shoulder = (normalized_shoulder_y - normalized_wrist_y) > self.min_wrist_above_shoulder
            
            # Detect shot release: wrist moving upward quickly AND above shoulder
            if (not self.is_shooting and 
                vertical_velocity < self.wrist_upward_velocity_threshold and
                self.min_wrist_height < normalized_wrist_y < self.max_wrist_height and
                wrist_above_shoulder):  # Key check to prevent false positives
                
                # Additional check: arm should be extending (elbow straightening)
                if len(self.arm_extension_history) >= 5:
                    recent_extension = self.arm_extension_history[-1]
                    earlier_extension = self.arm_extension_history[0]
                    extension_increase = recent_extension - earlier_extension
                    
                    # Check that arm is actually extending (not contracting)
                    if extension_increase > self.elbow_extension_threshold:
                        self.is_shooting = True
                        self.shot_detected = False
                        self.frames_since_release = 0
                        self.shooting_arm = arm_side
                        print(f"[ShotDetector] Shot detected! Arm: {arm_side}, Wrist velocity: {vertical_velocity:.4f}, Wrist above shoulder: {wrist_above_shoulder}")
            
            # Detect shot completion: wrist starts moving downward or returns to rest position
            elif (self.is_shooting and 
                  (vertical_velocity > 0.005 or  # Moving downward
                   normalized_wrist_y > 0.95 or  # Wrist too low
                   normalized_wrist_y < 0.1 or   # Wrist too high (unlikely but possible)
                   not wrist_above_shoulder)):  # Wrist dropped below shoulder
                
                if not self.shot_detected:
                    self._trigger_shot()
                    self.shot_detected = True
                    print(f"[ShotDetector] Shot release confirmed, starting 3-second recording")
        
        # Reset after shot completion
        if self.shot_detected:
            self.frames_since_release += 1
            if self.frames_since_release > self.max_frames_after_release:
                self.reset_state()
    
    def _get_shooting_arm_landmarks(self, landmarks: dict, height: int, width: int) -> Tuple:
        """Extract shooting arm landmarks (wrist, elbow, shoulder).
        
        Args:
            landmarks: MediaPipe pose landmarks dictionary
            height: Frame height
            width: Frame width
            
        Returns:
            Tuple of (wrist_pos, elbow_pos, shoulder_pos, arm_side) or (None, None, None, None)
        """
        # Try right arm first (most common)
        right_wrist = landmarks.get(self.RIGHT_WRIST)
        right_elbow = landmarks.get(self.RIGHT_ELBOW)
        right_shoulder = landmarks.get(self.RIGHT_SHOULDER)
        
        if (right_wrist and right_elbow and right_shoulder and
            right_wrist.get('visibility', 0) > self.min_confidence and
            right_elbow.get('visibility', 0) > self.min_confidence and
            right_shoulder.get('visibility', 0) > self.min_confidence):
            
            wrist_pos = (int(right_wrist['x'] * width), int(right_wrist['y'] * height))
            elbow_pos = (int(right_elbow['x'] * width), int(right_elbow['y'] * height))
            shoulder_pos = (int(right_shoulder['x'] * width), int(right_shoulder['y'] * height))
            return (wrist_pos, elbow_pos, shoulder_pos, 'right')
        
        # Try left arm
        left_wrist = landmarks.get(self.LEFT_WRIST)
        left_elbow = landmarks.get(self.LEFT_ELBOW)
        left_shoulder = landmarks.get(self.LEFT_SHOULDER)
        
        if (left_wrist and left_elbow and left_shoulder and
            left_wrist.get('visibility', 0) > self.min_confidence and
            left_elbow.get('visibility', 0) > self.min_confidence and
            left_shoulder.get('visibility', 0) > self.min_confidence):
            
            wrist_pos = (int(left_wrist['x'] * width), int(left_wrist['y'] * height))
            elbow_pos = (int(left_elbow['x'] * width), int(left_elbow['y'] * height))
            shoulder_pos = (int(left_shoulder['x'] * width), int(left_shoulder['y'] * height))
            return (wrist_pos, elbow_pos, shoulder_pos, 'left')
        
        return (None, None, None, None)
    
    def _calculate_velocity(self, positions: deque) -> Optional[Tuple[float, float]]:
        """Calculate velocity from position history.
        
        Args:
            positions: Deque of (x, y) position tuples
            
        Returns:
            (horizontal_velocity, vertical_velocity) or None
        """
        if len(positions) < 5:
            return None
            
        pos_list = list(positions)
        # Use first and last few positions for smoother velocity calculation
        dx = pos_list[-1][0] - pos_list[0][0]
        dy = pos_list[-1][1] - pos_list[0][1]
        dt = len(pos_list) - 1
        
        if dt == 0:
            return None
            
        return (dx / dt, dy / dt)
    
    def _trigger_shot(self):
        """Trigger shot detection callback."""
        if self.shot_callback:
            self.shot_callback()
    
    def get_last_ball_position(self) -> Optional[Tuple[int, int, float]]:
        """Get the last detected ball position (deprecated, kept for compatibility).
        
        Returns:
            None (ball tracking removed)
        """
        return None
    
    def reset_state(self):
        """Reset shot detection state."""
        self.is_shooting = False
        self.shot_detected = False
        self.frames_since_release = 0
        self.wrist_positions.clear()
        self.elbow_positions.clear()
        self.shoulder_positions.clear()
        self.arm_extension_history.clear()
        self.shooting_arm = None
