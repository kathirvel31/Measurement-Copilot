import argparse

from src.detect import detect_dress
from src.segment import segment_dress
from src.calibrate import calibrate
from src.keypoints import detect_keypoints
from src.measure import compute_all_measurements
from src.state import state
from config import config

def parse_args():
    parser = argparse.ArgumentParser(description="MECO Garment Measurement Co-Pilot")
    parser.add_argument("--image", "-i", default=config.get('defaults.test_img'), 
                        help="Path to garment image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def do_calibrate(source_img):
    print("📏 Step 0: Calibrating pixel-to-cm ratio using ArUco marker...")
    calibrate(source_img)
    if config.get('calibration.pixel_to_cm_ratio') is  None:
        print("⚠️ Calibration failed, cannot proceed without scale.")

def process_dress(source_img):
    print("⚖️ Calibration value:", config.get('calibration.pixel_to_cm_ratio'), "cm/px")
    # Update state with image
    state.update_img(source_img)
    # Ensure calibration is in state for measure.py to access
    state.update_calib(config.get('calibration.pixel_to_cm_ratio'), "ArUco", 0)

    print("🔍 Step 1: Detecting dresses...")
    detect_result = detect_dress(source_img)


    if len(detect_result):
        # Update state with detection results? 
        # Currently detect_dress returns YOLO results, not the dict format expected by state.update_det_result
        # But detect_keypoints internally updates state with keypoints AND detections from the keypoint model.
        
        print("✂️ Step 2: Segmenting dresses...")
        det_seg_result = segment_dress(detect_result, source_img)
        
        print("📍 Step 3: Detecting Keypoints...")
        keypoints_result = detect_keypoints(source_img)
        if keypoints_result:
             print(f"✅ Keypoints detected.")
             
             print("📏 Step 4: Computing Measurements...")
             measurements = compute_all_measurements()
             if measurements:
                 print(f"✅ Computed {len(measurements)} measurements.")
             else:
                 print("⚠️ No measurements computed.")

        else:
             print("⚠️ No keypoints detected.")

    else:
        print("⚠️ No dresses detected, skipping segmentation.")

    print("✅ Processing completed.")
    state.save_final_state()


if __name__ == "__main__":
    args = parse_args()
    source_img = args.image
    debug = args.debug

    if not config.get('preferences.optimizeCalibration'):
        do_calibrate(source_img)
    else:
        if config.get('calibration.pixel_to_cm_ratio') is  None:
            do_calibrate(source_img)

    if config.get('calibration.pixel_to_cm_ratio') is not None:
        process_dress(source_img)
    else:
        print("⚠️ Cannot proceed without calibration.")


