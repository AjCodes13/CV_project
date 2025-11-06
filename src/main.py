import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

from config import Config
from parking_space_detector import ParkingSpaceDetector
from feature_extractor import FeatureExtractor
from occupancy_classifier import OccupancyClassifier
from width_estimator import WidthEstimator

class ParkingSystem:
    """
    Main class to run the end-to-end parking analysis pipeline.
    """
    
    def __init__(self):
        self.config = Config()
        self.detector = ParkingSpaceDetector(self.config)
        self.extractor = FeatureExtractor(self.config)
        self.classifier = OccupancyClassifier(self.config)
        self.estimator = WidthEstimator(self.config)
        
        self.parking_spaces = self.detector.load_parking_spaces('parking_spaces.json')
        print(f"Loaded {len(self.parking_spaces)} parking spaces.")
        
    # def _initialize_estimator(self):
    #     """Initializes the WidthEstimator after calibration is loaded."""
    #     try:
    #         self.calibration.load_calibration('calibration.json')
    #         self.estimator = WidthEstimator(self.config, self.calibration)
    #         print("WidthEstimator initialized.")
    #     except FileNotFoundError:
    #         print("Error: calibration.json not found.")
    #         print("Please run the calibration script first.")
    #         exit()
    #     except Exception as e:
    #         print(f"Error initializing WidthEstimator: {e}")
    #         exit()

    def process_single_image(self, image_path: str, visualize=True):
        """
        Runs the full pipeline on a single image.
        """
        if self.estimator is None:
            self._initialize_estimator()
            
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
            
            features = self.extractor.extract_combined_features(roi)
            
            prediction, confidence = self.classifier.predict(features)
            occupancy_status = 'Occupied' if prediction == 1 else 'Empty'
            
            space_result = {
                'space_id': space_id,
                'occupancy': occupancy_status,
                'confidence': float(confidence),
                'parking_width': 0.0,
                'vehicle_width': 0.0,
                'assessment': None
            }

            parking_width_m = self.estimator.measure_parking_space_width(roi)
            space_result['parking_width'] = parking_width_m
            
            color = (0, 255, 0) 
            
            if occupancy_status == 'Occupied':
                color = (0, 0, 255) 
                
                has_vehicle, mask, contour_area = self.estimator.detect_vehicle_in_space(roi)
                if has_vehicle:
                    roi_area = roi.shape[0] * roi.shape[1]
                    occupancy_percentage = contour_area / roi_area

                    MIN_OCCUPANCY_THRESHOLD = 0.3

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
    
    TEST_IMAGE_PATH = "c:/Users/mhdda/parking_system/data/PKLot/UFPR05/Cloudy/2013-03-13/2013-03-13_07_25_01.jpg"
    MODEL_FILE = "c:/Users/mhdda/parking_system/models/svm_model.pkl"
    
    print("--- Starting Parking System ---")
    system = ParkingSystem()
    
    try:
        system.classifier.load_model(MODEL_FILE)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILE}' not found in 'models/' directory.")
        print("Please run the training script first.")
        exit()

    print(f"Processing image: {TEST_IMAGE_PATH}")
    results, result_image = system.process_single_image(TEST_IMAGE_PATH, visualize=True)
    
    for res in results:
        print(f"\n=== Space {res['space_id']} ===")
        print(f"  Status: {res['occupancy']} (Conf: {res['confidence']:.2f})")
        print(f"  Parking Width: {res['parking_width']:.2f}m")
        if res['occupancy'] == 'Occupied':
            print(f" Assessment: {res['assessment']['status']}")
            if res['assessment']:
                print(f" Offset: {res['assessment']['offset_m']:.2f}m")

    cv2.imshow("Parking Analysis Result", result_image)
    cv2.imwrite("data/results/parking_analysis_result.jpg", result_image)
    print("\nResult image saved to 'data/results/parking_analysis_result.jpg'")
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()