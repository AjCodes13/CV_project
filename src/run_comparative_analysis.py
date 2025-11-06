# # src/run_comparative_analysis.py

# import cv2
# import numpy as np
# from pathlib import Path
# import sys
# import xml.etree.ElementTree as ET

# # --- Imports will work directly, since we are in the src/ folder ---
# from config import Config
# from parking_space_detector import ParkingSpaceDetector
# from width_estimator import WidthEstimator
# # -------------------------------------------------------------------

# # --- NEW HEURISTIC (RULE-BASED) CLASSIFIER ---
# def classify_heuristic_canny(roi_image, threshold):
#     """
#     Classifies a space as "Occupied" or "Empty" based on
#     a simple Canny edge-counting heuristic.
#     """
#     # 1. Convert to grayscale and blur to reduce noise
#     gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # 2. Find edges
#     edges = cv2.Canny(blurred, 50, 150)
    
#     # 3. Count non-zero pixels (the edges)
#     edge_count = cv2.countNonZero(edges)
    
#     # 4. Apply the "magic number" threshold
#     if edge_count > threshold:
#         return "Occupied", edge_count
#     else:
#         return "Empty", edge_count
# # -----------------------------------------------

# class ParkingSystemHeuristic:
#     """
#     A modified ParkingSystem that uses the heuristic classifier
#     instead of the SVM.
#     """
    
#     def __init__(self):
#         self.config = Config()
#         self.detector = ParkingSpaceDetector(self.config)
#         self.estimator = WidthEstimator(self.config)
        
#         # Load the predefined parking spaces
#         self.parking_spaces = self.detector.load_parking_spaces('parking_spaces.json')
#         print(f"Loaded {len(self.parking_spaces)} parking spaces.")

#     def process_single_image(self, image_path: str, heuristic_threshold: int, visualize=True):
#         """
#         Runs the full pipeline on a single image.
#         """
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Error: Could not load image {image_path}")
#             return [], None
            
#         result_image = image.copy()
#         results = []

#         for space in self.parking_spaces:
#             space_id = space['id']
#             corners = np.array(space['corners'])
            
#             # 1. Extract ROI (still needed)
#             roi = self.detector.extract_parking_roi(image, space)
            
#             # 2. --- CLASSIFY (Using Heuristic) ---
#             occupancy_status, edge_val = classify_heuristic_canny(roi, heuristic_threshold)
            
#             space_result = {
#                 'space_id': space_id,
#                 'occupancy': occupancy_status,
#                 'confidence': edge_val, # We'll store the edge_count here
#                 'parking_width': 0.0,
#                 'vehicle_width': 0.0,
#                 'assessment': None
#             }

#             # 3. Measure Parking Width (as before)
#             parking_width_m = self.estimator.measure_parking_space_width(roi)
#             space_result['parking_width'] = parking_width_m
            
#             color = (0, 255, 0) # Green for Empty
            
#             if occupancy_status == 'Occupied':
#                 color = (0, 0, 255) # Red for Occupied
                
#                 # 4. Detect vehicle in the *top-down ROI* (as before)
#                 has_vehicle, mask, contour_area = \
#                     self.estimator.detect_vehicle_in_space(roi)

#                 # 5. --- 50% OCCUPANCY RULE (as before) ---
#                 if has_vehicle:
#                     roi_area = roi.shape[0] * roi.shape[1]
#                     occupancy_percentage = contour_area / roi_area
                    
#                     MIN_OCCUPANCY_THRESHOLD = 0.4 
                    
#                     if occupancy_percentage < MIN_OCCUPANCY_THRESHOLD:
#                         occupancy_status = "Empty"
#                         space_result['occupancy'] = "Empty"
#                         color = (0, 255, 0) 
#                         has_vehicle = False
                
#                 # 6. --- Centering Assessment (as before) ---
#                 if has_vehicle:
#                     assessment, offset = self.estimator.assess_centering(roi, mask)
#                     space_result['assessment'] = assessment
#                     space_result['vehicle_width'] = 0.0 
            
#             results.append(space_result)

#             # 7. Visualize (as before)
#             if visualize:
#                 cv2.polylines(result_image, [corners.astype(np.int32)], True, color, 2)
#                 text_pos = (corners[0][0], corners[0][1] - 10)
#                 cv2.putText(result_image, f"{space_id}: {occupancy_status}", 
#                             text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
#                 if occupancy_status == 'Occupied' and space_result['assessment']:
#                      cv2.putText(result_image, f"Fit: {space_result['assessment']['status']}", 
#                             (text_pos[0], text_pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

#         return results, result_image

# # --- This part runs the script ---
# if __name__ == "__main__":
    
#     # --- 1. CONFIGURE THIS ---
#     # Use the *same* test image as your main.py for a good comparison
#     TEST_IMAGE_PATH = "c:/Users/mhdda/parking_system/data/PKLot/UFPR05/Cloudy/2013-03-13/2013-03-13_07_20_01.jpg"
    
#     # This is the "magic number". You will have to TUNE this value
#     HEURISTIC_THRESHOLD = 1500
    
#     # ---------------------------
    
#     print("--- Starting Heuristic Analysis System ---")
#     system = ParkingSystemHeuristic()

#     print(f"Processing image: {TEST_IMAGE_PATH}")
#     print(f"Using edge threshold: {HEURISTIC_THRESHOLD}")
    
#     results, result_image = system.process_single_image(TEST_IMAGE_PATH, 
#                                                         HEURISTIC_THRESHOLD, 
#                                                         visualize=True)
    
#     # 4. Print results
#     for res in results:
#         print(f"\n=== Space {res['space_id']} ===") # Changed to res['id']
#         print(f"  Status: {res['occupancy']} (Edge Count: {res['confidence']})")
#         print(f"  Parking Width: {res['parking_width']:.2f}m")
#         if res['occupancy'] == 'Occupied' and res['assessment']:
#             print(f"  Assessment: {res['assessment']['status']}")
#             print(f"  Offset: {res['assessment']['offset_m']:.2f}m")

#     # 5. Show and save the result image
#     output_filename = f"data/results/heuristic_result_thresh_{HEURISTIC_THRESHOLD}.jpg"
#     cv2.imshow("Heuristic Analysis Result", result_image)
#     cv2.imwrite(output_filename, result_image)
#     print(f"\nResult image saved to {output_filename}")
    
#     print("Press any key to exit.")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np
from pathlib import Path
import sys

from config import Config
from parking_space_detector import ParkingSpaceDetector
from width_estimator import WidthEstimator
from feature_extractor import FeatureExtractor

class ParkingSystemHeuristic:
    
    def __init__(self):
        self.config = Config()
        self.detector = ParkingSpaceDetector(self.config)
        self.estimator = WidthEstimator(self.config)
        self.extractor = FeatureExtractor(self.config) 
        
        self.parking_spaces = self.detector.load_parking_spaces('parking_spaces.json')
        print(f"Loaded {len(self.parking_spaces)} parking spaces.")
        
        try:
            ref_path = self.config.PROCESSED_DIR / "reference_lbp_histogram.npy"
            self.reference_histogram = np.load(ref_path)
            print("Loaded LBP reference histogram for comparison.")
        except FileNotFoundError:
            print("Error: reference_lbp_histogram.npy not found.")
            print("Please run 'run_create_reference.py' first.")
            sys.exit()

    def classify_heuristic_lbp(self, roi_image, threshold):
        """
        Classifies a space by comparing its LBP histogram to the reference.
        """
        current_histogram = self.extractor.extract_lbp_features(roi_image)
        
        hist1 = self.reference_histogram.astype(np.float32)
        hist2 = current_histogram.astype(np.float32)

        distance = cv2.compareHist(hist1, 
                                hist2, 
                                cv2.HISTCMP_CHISQR)
        
        if distance > threshold:
            return "Occupied", distance
        else:
            return "Empty", distance
    def process_single_image(self, image_path: str, heuristic_threshold: int, visualize=True):
        """
        Runs the full pipeline on a single image.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return [], None
            
        result_image = image.copy()
        results = []

        for space in self.parking_spaces:
            space_id = space['id']
            corners = np.array(space['corners'])
            
            roi = self.detector.extract_parking_roi(image, space)
            
            occupancy_status, distance_score = self.classify_heuristic_lbp(roi, heuristic_threshold)
            
            space_result = {
                'space_id': space_id,
                'occupancy': occupancy_status,
                'confidence': distance_score, 
                'parking_width': 0.0,
                'vehicle_width': 0.0,
                'assessment': None
            }

            parking_width_m = self.estimator.measure_parking_space_width(roi)
            space_result['parking_width'] = parking_width_m
            
            color = (0, 255, 0)
            
            if occupancy_status == 'Occupied':
                color = (0, 0, 255)
                
                has_vehicle, mask, contour_area = \
                    self.estimator.detect_vehicle_in_space(roi)

                if has_vehicle:
                    roi_area = roi.shape[0] * roi.shape[1]
                    occupancy_percentage = contour_area / roi_area
                    
                    MIN_OCCUPANCY_THRESHOLD = 0.4 
                    
                    if occupancy_percentage < MIN_OCCUPANCY_THRESHOLD:
                        occupancy_status = "Empty"
                        space_result['occupancy'] = "Empty"
                        color = (0, 255, 0) 
                        has_vehicle = False
                
                if has_vehicle:
                    assessment, offset = self.estimator.assess_centering(roi, mask)
                    space_result['assessment'] = assessment
                    space_result['vehicle_width'] = 0.0 
            
            results.append(space_result)

            if visualize:
                cv2.polylines(result_image, [corners.astype(np.int32)], True, color, 2)
                text_pos = (corners[0][0], corners[0][1] - 10)
                cv2.putText(result_image, f"{space_id}: {occupancy_status}", 
                            text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if occupancy_status == 'Occupied' and space_result['assessment']:
                     cv2.putText(result_image, f"Fit: {space_result['assessment']['status']}", 
                            (text_pos[0], text_pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        return results, result_image
if __name__ == "__main__":
    
    TEST_IMAGE_PATH = "c:/Users/mhdda/parking_system/data/PKLot/UFPR05/Cloudy/2013-03-13/2013-03-13_07_20_01.jpg"
    
    HEURISTIC_THRESHOLD = 0.07
    
    print("--- Starting Heuristic Analysis System ---")
    system = ParkingSystemHeuristic()

    print(f"Processing image: {TEST_IMAGE_PATH}")
    print(f"Using edge threshold: {HEURISTIC_THRESHOLD}")
    
    results, result_image = system.process_single_image(TEST_IMAGE_PATH, 
                                                        HEURISTIC_THRESHOLD, 
                                                        visualize=True)
    
    for res in results:
        print(f"\n=== Space {res['space_id']} ===") 
        print(f"  Status: {res['occupancy']} (Distance Score: {res['confidence']:.2f})")
        print(f"  Parking Width: {res['parking_width']:.2f}m")
        if res['occupancy'] == 'Occupied' and res['assessment']:
            print(f"  Assessment: {res['assessment']['status']}")
            print(f"  Offset: {res['assessment']['offset_m']:.2f}m")

    output_filename = f"data/results/heuristic_result_thresh_{HEURISTIC_THRESHOLD}.jpg"
    cv2.imshow("Heuristic Analysis Result", result_image)
    cv2.imwrite(output_filename, result_image)
    print(f"\nResult image saved to {output_filename}")
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()