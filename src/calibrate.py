import cv2
import numpy as np
import os
from config import config

# Dict name to OpenCV ID mapping
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

def calibrate(image_path):
    """
    Calibrate pixel-to-cm ratio using ArUco markers.
    """
    marker_size_cm = config.get('calibration.marker_size_cm', 5.0)
    aruco_dict_name = config.get('calibration.aruco_dict', 'DICT_5X5_50')
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return None

    # Set up ArUco detection
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS.get(aruco_dict_name, cv2.aruco.DICT_5X5_50))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(img)

    if ids is not None and len(ids) > 0:
        # We take the first detected marker for calibration
        marker_corners = corners[0][0]
        
        # Calculate pixel perimeter or side average
        # Side lengths in pixels
        side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
        side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
        side3 = np.linalg.norm(marker_corners[2] - marker_corners[3])
        side4 = np.linalg.norm(marker_corners[3] - marker_corners[0])
        
        avg_pixel_side = (side1 + side2 + side3 + side4) / 4.0
        
        # pixel_to_cm_ratio = physical_size / pixel_size
        pixel_to_cm_ratio = marker_size_cm / avg_pixel_side
        
            
        config.update_calibration(pixel_to_cm_ratio)
        
        print(f"✅ ArUco marker detected with ID: {ids[0][0]}")
        print(f"📏 Average pixel side: {avg_pixel_side:.2f}px")
        
        # --- Visualization ---
        viz_img = img.copy()
        cv2.aruco.drawDetectedMarkers(viz_img, corners, ids)
        
        # Add ratio text to image
        text = f"Ratio: {pixel_to_cm_ratio:.6f} cm/px"
        cv2.putText(viz_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Create output directory: runs/calibration/YYYYMMDD_HHMMSS/
        timestamp = config.get('calibration.calibrated_at').replace(':', '-').replace('.', '-')
        # Simplified timestamp for folder name
        from datetime import datetime
        folder_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.get('paths.output', 'runs'), 'calibration', folder_timestamp)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        save_path = os.path.join(output_dir, "calibration_result.jpg")
        cv2.imwrite(save_path, viz_img)
        print(f"🖼️ Calibration visualization saved to: {save_path}")
        
        return {
            'id': ids[0][0],
            'corners': marker_corners,
            'pixel_to_cm_ratio': pixel_to_cm_ratio,
            'viz_path': save_path
        }
    else:
        print("⚠️ No ArUco markers detected.")
        config.update_calibration(None)
        return None
