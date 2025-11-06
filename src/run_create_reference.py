import cv2
import numpy as np
from pathlib import Path

from config import Config
from parking_space_detector import ParkingSpaceDetector
from feature_extractor import FeatureExtractor

REFERENCE_IMAGE_PATH = "C:/Users/mhdda/parking_system/data/PKLot/UFPR05/Cloudy/2013-03-13/2013-03-13_06_40_00.jpg"
REFERENCE_SPACE_ID = 16 

def main():
    print("--- Creating 'No-ML' Reference Histogram ---")
    config = Config()
    detector = ParkingSpaceDetector(config)
    extractor = FeatureExtractor(config)

    spaces = detector.load_parking_spaces('parking_spaces.json')
    
    ref_space = next((s for s in spaces if s['id'] == REFERENCE_SPACE_ID), None)
    if ref_space is None:
        print(f"Error: Space ID {REFERENCE_SPACE_ID} not found.")
        return

    image = cv2.imread(REFERENCE_IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image {REFERENCE_IMAGE_PATH}")
        return
        
    roi = detector.extract_parking_roi(image, ref_space)
    
    reference_histogram = extractor.extract_lbp_features(roi)
    
    output_path = config.PROCESSED_DIR / "reference_lbp_histogram.npy"
    np.save(output_path, reference_histogram)
    
    print(f"Successfully saved LBP reference histogram to:")
    print(output_path)
    print(f"Reference shape: {reference_histogram.shape}")

if __name__ == "__main__":
    main()