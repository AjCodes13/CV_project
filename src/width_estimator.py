import cv2
import numpy as np
import random

class WidthEstimator:
    """
    Measures vehicle widths and assesses fit based on
    standardized, top-down ROIs.
    """

    def __init__(self, config):
        self.config = config

    def measure_parking_space_width(self, roi: np.ndarray) -> float:
        """
        The parking space width is now a known constant.
        """
        return self.config.STANDARD_PARKING_WIDTH

    def detect_vehicle_in_space(self, roi: np.ndarray) -> (bool, np.ndarray, float):
        """
        Detects the vehicle within the ROI using thresholding and contours.
        Returns (has_vehicle, vehicle_mask, contour_area)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        otsu_thresh_val, car_shadow_mask = cv2.threshold(gray, 0, 255, 
                                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        padding_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

        final_car_mask = cv2.bitwise_and(car_shadow_mask, padding_mask)

        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(final_car_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        if contour_area < 1000: 
            return False, None, 0.0

        vehicle_mask = np.zeros_like(clean_mask)
        cv2.drawContours(vehicle_mask, [largest_contour], -1, (255), -1)

        return True, vehicle_mask, contour_area

    # def measure_vehicle_width(self, roi: np.ndarray, mask: np.ndarray) -> (float, tuple):
    #     """
    #     Measures the width of the vehicle from its mask.
    #     The mask is from a standardized ROI, so we can use
    #     our new pixel-per-meter constant.
    #     """
    #     points = cv2.findNonZero(mask)

    #     if points is None:
    #         return 0.0, None

    #     # Get the bounding box
    #     x, y, w, h = cv2.boundingRect(points)

    #     pixel_width = w

    #     # Convert pixel width to meters using our new constant
    #     width_in_meters = pixel_width / self.config.ROI_PIXELS_PER_METER

    #     # For visualization
    #     left_point = (x, y + h // 2)
    #     right_point = (x + w, y + h // 2)

    #     return width_in_meters, (left_point, right_point)

    # def assess_parking_fit(self, vehicle_width: float, parking_width: float) -> dict:
    #     """
    #     Assesses if the vehicle fits based on clearance.
    #     (This function remains unchanged)
    #     """
    #     # if vehicle_width >= parking_width:
    #     #     random_subtraction = random.uniform(0.3, 0.5)
    #     #     vehicle_width = parking_width - random_subtraction

    #     assessment = {
    #         'can_fit': False,
    #         'status': 'N/A',
    #         'total_clearance': 0.0,
    #         'clearance_each_side': 0.0
    #     }

    #     if vehicle_width <= 0 or parking_width <= 0 or vehicle_width > parking_width:
    #         assessment['status'] = "Measurement Error"
    #         return assessment

    #     total_clearance = parking_width - vehicle_width
    #     clearance_each_side = total_clearance / 2.0

    #     assessment['total_clearance'] = total_clearance
    #     assessment['clearance_each_side'] = clearance_each_side

    #     if total_clearance < 0:
    #         assessment['status'] = "Too Wide"
    #     elif clearance_each_side < self.config.REQUIRED_CLEARANCE:
    #         assessment['status'] = "Very Tight"
    #         assessment['can_fit'] = True
    #     elif clearance_each_side < (self.config.REQUIRED_CLEARANCE * 2):
    #         assessment['status'] = "Good Fit"
    #         assessment['can_fit'] = True
    #     else:
    #         assessment['status'] = "Excellent Fit"
    #         assessment['can_fit'] = True

    #     return assessment
    def assess_centering(self, roi: np.ndarray, vehicle_mask: np.ndarray) -> (dict, float):
        """
        Assesses how well the car is centered within the parking space's width.
        Returns: (assessment_dict, offset_in_meters)
        """
        assessment = {
            'status': 'N/A',
            'offset_m': 0.0
        }

        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            assessment['status'] = "Centering Error"
            return assessment, 0.0

        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            assessment['status'] = "Centering Error"
            return assessment, 0.0

        car_center_x_px = int(M["m10"] / M["m00"])

        space_center_x_px = roi.shape[1] / 2

        pixel_offset = abs(car_center_x_px - space_center_x_px)

        offset_m = pixel_offset / self.config.ROI_PIXELS_PER_METER
        assessment['offset_m'] = round(offset_m, 2)

        if offset_m < 0.15:  
            assessment['status'] = "Well Centered"
        elif offset_m < 0.3: 
            assessment['status'] = "Slightly Off-Center"
        else:
            assessment['status'] = "Poorly Centered"

        return assessment, offset_m