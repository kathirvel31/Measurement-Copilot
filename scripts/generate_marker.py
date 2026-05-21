import cv2
import numpy as np
import os

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

def generate_marker(marker_id=0, dict_name="DICT_4X4_50", size=400, output_path="marker.png"):
    """
    Generate an ArUco marker image.
    """
    if dict_name not in ARUCO_DICTS:
        print(f"❌ Error: {dict_name} is not a valid dictionary name.")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    
    # Generate marker
    marker_img = np.zeros((size, size), dtype=np.uint8)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    
    # Save marker
    cv2.imwrite(output_path, marker_img)
    print(f"✅ Marker saved to {output_path}")
    print(f"📄 Dictionary: {dict_name}, ID: {marker_id}")

if __name__ == "__main__":
    if not os.path.exists("scripts"):
        os.makedirs("scripts")
    generate_marker(output_path="scripts/aruco_marker_0.png")
