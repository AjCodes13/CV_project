import cv2
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config import Config
from calibration import ParkingCalibration

REFERENCE_IMAGE_PATH = "C:/Users/mhdda/parking_system/data/PKLot/UFPR05/Sunny/2013-03-02/2013-03-02_06_45_00.jpg"

OUTPUT_JSON_FILE = "calibration.json"


def main():
    print("--- Parking Space Calibrator ---")
    
    config = Config()
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    image_path = Path(REFERENCE_IMAGE_PATH)
    if not image_path.exists():
        print(f"Error: Reference image not found at {REFERENCE_IMAGE_PATH}")
        print("Please update REFERENCE_IMAGE_PATH in this script.")
        return
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
        
    print(f"Loaded reference image: {REFERENCE_IMAGE_PATH}")
    
    print("\nStarting manual calibration...")
    calibrator = ParkingCalibration(config)
    calibration_data = calibrator.manual_calibration(image)
    
    if not calibration_data:
        print("Calibration was cancelled. Exiting.")
        return
        
    print(f"\nCalibration complete.")
    print(f"Data: {calibration_data}")
    print(f"Successfully saved calibration to data/processed/{OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    main()
