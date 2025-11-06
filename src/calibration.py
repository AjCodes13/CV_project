import cv2
import numpy as np
import json
from pathlib import Path

class ParkingCalibration:
    """
    Handles camera calibration to find the pixels-per-meter ratio.
    """

    def __init__(self, config):
        self.config = config
        self.calibration_data = {}
        self.points = []

    def _mouse_callback(self, event, x, y, flags, param):
        """Internal mouse callback for manual calibration."""
        img_copy = param['image'].copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                
            if len(self.points) == 2:
                cv2.line(img_copy, self.points[0], self.points[1], (0, 0, 255), 2)
                
        cv2.imshow("Calibration: Click 2 points", img_copy)

    def manual_calibration(self, image: np.ndarray):
        """
        Interactively calibrate by clicking two points across a known distance
        (the standard parking width).
        """
        print(f"Calibrating using standard width: {self.config.STANDARD_PARKING_WIDTH} meters")
        print("Please click the two points spanning ONE parking space width.")
        print("Press 's' to save and exit, 'r' to reset, 'q' to quit.")
        
        cv2.imshow("Calibration: Click 2 points", image)
        cv2.setMouseCallback("Calibration: Click 2 points", self._mouse_callback, {'image': image})
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if len(self.points) == 2:
                    pixel_distance = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1]))
                    pixels_per_meter = pixel_distance / self.config.STANDARD_PARKING_WIDTH
                    self.calibration_data = {'pixels_per_meter': pixels_per_meter}
                    print(f"Pixel distance: {pixel_distance:.2f}")
                    print(f"Pixels-per-meter: {pixels_per_meter:.2f}")
                    self.save_calibration('calibration.json')
                    break
                else:
                    print("Error: Please select exactly two points.")
                    
            elif key == ord('r'):
                self.points = []
                cv2.imshow("Calibration: Click 2 points", image)
                print("Points reset. Please select two points.")
                
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return self.calibration_data

    def save_calibration(self, filename: str):
        """Save calibration data to JSON."""
        filepath = self.config.PROCESSED_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(self.calibration_data, f, indent=4)
        print(f"Calibration data saved to {filepath}")

    def load_calibration(self, filename: str):
        """Load calibration data from JSON."""
        filepath = self.config.PROCESSED_DIR / filename
        with open(filepath, 'r') as f:
            self.calibration_data = json.load(f)
        print(f"Calibration data loaded from {filepath}")
        return self.calibration_data

    def get_pixels_per_meter(self) -> float:
        return self.calibration_data.get('pixels_per_meter')