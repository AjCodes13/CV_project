import cv2
import numpy as np
import json
from typing import List, Tuple, Dict

class ParkingSpaceDetector:
    """
    Detects and defines parking space regions of interest (ROIs)
    Uses edge detection and line detection for automated detection,
    with manual annotation fallback
    """
    
    def __init__(self, config):
        self.config = config
        self.parking_spaces = []
        
    def detect_parking_lines(self, image: np.ndarray) -> List[Tuple]:
        """
        Detect parking space boundary lines using Hough Transform
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 
                         self.config.CANNY_THRESHOLD1,
                         self.config.CANNY_THRESHOLD2)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        lines = cv2.HoughLinesP(edges,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.config.HOUGH_THRESHOLD,
                               minLineLength=self.config.HOUGH_MIN_LINE_LENGTH,
                               maxLineGap=self.config.HOUGH_MAX_LINE_GAP)
        
        return lines if lines is not None else []
    
    def filter_parking_lines(self, lines: List[Tuple], 
                            image_shape: Tuple) -> List[Tuple]:
        """
        Filter lines to keep only parking space boundaries
        (vertical lines for perpendicular parking)
        """
        filtered_lines = []
        height, width = image_shape[:2]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if 70 <= angle <= 110 or angle >= 170 or angle <= 10:
                filtered_lines.append(line)
        
        return filtered_lines
    
    def define_parking_spaces_manual(self, image: np.ndarray) -> List[Dict]:
        """
        Manual ROI definition using mouse clicks
        Click 4 corners of each parking space
        """
        spaces = []
        current_points = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points
            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append((x, y))
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Define Parking Spaces", img_copy)
                
                if len(current_points) == 4:
                    # Draw the parking space
                    pts = np.array(current_points, np.int32)
                    cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
                    cv2.imshow("Define Parking Spaces", img_copy)
                    
                    spaces.append({
                        'id': len(spaces),
                        'corners': current_points.copy()
                    })
                    current_points = []
        
        img_copy = image.copy()
        cv2.imshow("Define Parking Spaces", img_copy)
        cv2.setMouseCallback("Define Parking Spaces", mouse_callback)
        
        print("Click 4 corners of each parking space")
        print("Press 's' to save, 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return spaces
    
    def save_parking_spaces(self, spaces: List[Dict], filename: str):
        """Save parking space definitions to JSON"""
        filepath = self.config.PROCESSED_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(spaces, f, indent=4)
    
    def load_parking_spaces(self, filename: str) -> List[Dict]:
        """Load parking space definitions from JSON"""
        filepath = self.config.PROCESSED_DIR / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_parking_roi(self, image: np.ndarray, 
                          space: Dict) -> np.ndarray:
        """
        Extracts and un-warps the ROI for a single parking space
        to a standardized, top-down perspective.
        """
        corners = np.array(space['corners'], dtype=np.float32)

        output_width_px = int(self.config.STANDARD_PARKING_WIDTH * self.config.ROI_PIXELS_PER_METER)
        output_height_px = int(self.config.STANDARD_PARKING_LENGTH * self.config.ROI_PIXELS_PER_METER)

        dst_corners = np.array([
            [0, 0],
            [output_width_px - 1, 0],
            [output_width_px - 1, output_height_px - 1],
            [0, output_height_px - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst_corners)
        roi = cv2.warpPerspective(image, M, (output_width_px, output_height_px))

        return roi
    def get_perspective_transform(self, space: Dict) -> (np.ndarray, int):
        """
        Calculates the perspective transform matrix (M) for a space.
        Returns: (M, output_width_in_pixels)
        """
        corners = np.array(space['corners'], dtype=np.float32)

        output_width_px = int(self.config.STANDARD_PARKING_WIDTH * self.config.ROI_PIXELS_PER_METER)
        output_height_px = int(self.config.STANDARD_PARKING_LENGTH * self.config.ROI_PIXELS_PER_METER)

        dst_corners = np.array([
            [0, 0],
            [output_width_px - 1, 0],
            [output_width_px - 1, output_height_px - 1],
            [0, output_height_px - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst_corners)
        return M, output_width_px