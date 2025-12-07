"""Ball detection and tracking for shot detection."""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class BallTracker:
    """Tracks basketball using color-based detection."""
    
    def __init__(self):
        """Initialize ball tracker."""
        # Orange basketball color range (HSV)
        # Lower and upper bounds for orange color - expanded range for better detection
        self.lower_orange = np.array([0, 50, 50])  # More lenient lower bound
        self.upper_orange = np.array([30, 255, 255])  # Extended upper bound
        
        # Alternative orange range for different lighting
        self.lower_orange2 = np.array([5, 100, 100])
        self.upper_orange2 = np.array([25, 255, 255])
        
        # Minimum ball radius (in pixels)
        self.min_radius = 5  # Reduced minimum for smaller balls in video
        self.max_radius = 200  # Increased maximum for closer shots
        
        # Tracking state
        self.last_ball_position: Optional[Tuple[int, int]] = None
        self.ball_velocity: Optional[Tuple[float, float]] = None
        self.frames_without_ball = 0
        self.last_contour: Optional[np.ndarray] = None  # Store contour for skeleton drawing
        
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Detect basketball in frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            (x, y, radius) of detected ball or None
        """
        if frame is None:
            return None
        
        # Ensure frame is in correct format (BGR, uint8)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Ensure frame has 3 channels
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return None
            
        # Validate frame before color conversion
        if frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
            return None
        
        # Convert to HSV
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(f"[BallTracker] ERROR in HSV conversion: {e}, frame shape: {frame.shape if frame is not None else 'None'}")
            import sys
            sys.stdout.flush()
            return None
        
        # Create mask for orange color - try both ranges for better detection
        try:
            mask1 = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
            mask2 = cv2.inRange(hsv, self.lower_orange2, self.upper_orange2)
            mask = cv2.bitwise_or(mask1, mask2)  # Combine both masks
        except Exception as e:
            print(f"[BallTracker] ERROR creating mask: {e}")
            import sys
            sys.stdout.flush()
            return None
        
        # Apply morphological operations to reduce noise
        try:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            print(f"[BallTracker] ERROR in morphological operations: {e}")
            import sys
            sys.stdout.flush()
            return None
        
        # Find contours
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            print(f"[BallTracker] ERROR finding contours: {e}")
            import sys
            sys.stdout.flush()
            return None
        
        # Debug: Log mask statistics periodically
        if hasattr(self, 'frames_without_ball') and self.frames_without_ball % 90 == 0 and self.frames_without_ball > 0:
            try:
                mask_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                mask_percentage = (mask_pixels / total_pixels) * 100
                print(f"[BallTracker] Mask: {mask_pixels}/{total_pixels} pixels ({mask_percentage:.2f}%), {len(contours)} contours found")
                
                # Log top 3 contours by area
                if contours:
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
                    for i, contour in enumerate(sorted_contours):
                        try:
                            area = cv2.contourArea(contour)
                            if area > 0:
                                (cx, cy), r = cv2.minEnclosingCircle(contour)
                                perimeter = cv2.arcLength(contour, True)
                                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                                print(f"[BallTracker]   Contour {i+1}: area={area:.1f}, radius={int(r)}, circularity={circularity:.3f}, center=({int(cx)}, {int(cy)})")
                        except Exception as e:
                            print(f"[BallTracker]   Error processing contour {i+1}: {e}")
                
                import sys
                sys.stdout.flush()
            except Exception as e:
                print(f"[BallTracker] Error in debug logging: {e}")
                import sys
                sys.stdout.flush()
        
        if not contours:
            self.frames_without_ball += 1
            self.last_contour = None
            return None
        
        # Sort contours by area (largest first) and check each one
        # This allows us to find the best match, not just the largest
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Use a scoring system to find the best ball candidate
        best_ball = None
        best_score = 0
        
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (circular objects) - minimum threshold
            if area < 100:  # Increased from 50 to reduce false positives
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)
            
            # Filter by radius - reasonable range for basketball
            max_radius = 250  # Reduced from 300
            if radius < self.min_radius or radius > max_radius:
                continue
            
            # Check circularity - need reasonable circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Require minimum circularity of 0.4 (was 0.3, too low)
            if circularity < 0.4:
                continue
            
            # Additional check: area should be close to circle area (pi * r^2)
            # This helps filter out non-circular shapes
            expected_area = np.pi * radius * radius
            area_ratio = area / expected_area if expected_area > 0 else 0
            
            # Area should be at least 60% of the enclosing circle (for a solid ball)
            # and not more than 100% (shouldn't happen, but safety check)
            if area_ratio < 0.6 or area_ratio > 1.1:
                continue
            
            # Calculate a score: higher is better
            # Prefer: higher circularity, area ratio closer to 1.0, reasonable size
            score = (circularity * 0.5) + (min(area_ratio, 1.0) * 0.3) + (min(radius / 100.0, 1.0) * 0.2)
            
            # Only accept if score is above threshold
            if score > 0.5 and score > best_score:
                best_ball = (x, y, radius, contour, score)
                best_score = score
        
        # If we found a good ball candidate
        if best_ball is not None:
            x, y, radius, contour, score = best_ball
            self.last_ball_position = (x, y)
            self.last_contour = contour
            self.frames_without_ball = 0
            
            # Log successful detection
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            print(f"[BallTracker] Ball detected! x={x}, y={y}, radius={radius}, area={area:.1f}, circularity={circularity:.3f}, score={score:.3f}")
            import sys
            sys.stdout.flush()
            
            return (x, y, radius)
        
        # No valid ball found in any contour
        if self.frames_without_ball % 90 == 0:
            print(f"[BallTracker] No valid ball found in {len(contours)} contours")
            import sys
            sys.stdout.flush()
        self.frames_without_ball += 1
        self.last_contour = None
        return None
        
    
    def draw_ball(self, frame: np.ndarray, ball_pos: Optional[Tuple[int, int, float]]):
        """Draw ball detection on frame.
        
        Args:
            frame: Frame to draw on
            ball_pos: (x, y, radius) of ball or None
        """
        if ball_pos is not None:
            x, y, radius = ball_pos
            # Draw contour/skeleton if available
            if self.last_contour is not None:
                cv2.drawContours(frame, [self.last_contour], -1, (0, 255, 255), 2)
            
            # Draw enclosing circle
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    
    def get_last_contour(self) -> Optional[np.ndarray]:
        """Get the last detected ball contour.
        
        Returns:
            Contour array or None
        """
        return self.last_contour

