
""" visualize from  the final result .. """
import cv2
import numpy as np
import json
from pathlib import Path
import sys

POINT_COLOR = (0, 0, 255)  #RED points


GROUP_COLORS = [
    (255, 0, 0),    #BLUE - Group 1
    (0, 255, 0),    #GREEN - Group 2  
    (0, 165, 255),  #ORANGE - Group 3
    (255, 0, 255),  #MAGENTA - Group 4
    (255, 165, 0),  #GOLD - Group 5
]

def load_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    return img

def draw_keypoint(img: np.ndarray, kp: list, kp_id: int, size=8):
    x, y = int(kp[0]), int(kp[1])
    cv2.circle(img, (x, y), size, POINT_COLOR, -1)
    cv2.circle(img, (x, y), size//2, (0, 0, 0), 2)
    cv2.putText(img, str(kp_id), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, POINT_COLOR, 2)
    return img

def draw_line_only(img: np.ndarray, kp1: list, kp2: list, color, thickness=3):
    """Draw line WITHOUT text - for overlaps use thicker lines"""
    x1, y1 = int(kp1[0]), int(kp1[1])
    x2, y2 = int(kp2[0]), int(kp2[1])
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.arrowedLine(img, (mid_x-25, mid_y), (mid_x+25, mid_y), color, 2)
    return img

def draw_line_with_text(img: np.ndarray, kp1: list, kp2: list, text: str, color, thickness=3):
    """Draw line WITH text - used ONCE per group"""
    x1, y1 = int(kp1[0]), int(kp1[1])
    x2, y2 = int(kp2[0]), int(kp2[1])
    
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.arrowedLine(img, (mid_x-25, mid_y), (mid_x+25, mid_y), color, 2)
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(img, (mid_x - text_size[0]//2 - 8, mid_y - text_size[1] - 8),
                  (mid_x + text_size[0]//2 + 8, mid_y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (mid_x - text_size[0]//2, mid_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def visualize_result(result_json_path: str):
    print(f"🎨 Loading: {result_json_path}")
    
    with open(result_json_path, 'r') as f:
        state_data = json.load(f)
    
    img_path = state_data['img_path']
    img = load_image(img_path)
    result_folder = Path(result_json_path).parent
    
    # for All 14 keypoints-->> Red
    keypoints = state_data['keypoints_result']['detections'][0]['keypoints']
    for i, kp in enumerate(keypoints):
        if kp[0] > 0 and kp[1] > 0:
            img = draw_keypoint(img, kp, i)
    
    #EACH GROUP = UNIQUE COLOR + 1 TEXT ONLY
    measurements = state_data['measurements_result']
    print(f"📏 {len(measurements)} measurement groups:")
    
    for group_idx, meas in enumerate(measurements):
        meas_name = meas['measurement_pair']
        value = meas['value']
        unit = meas['unit']
        group_keypoints = meas['keypoints']
        color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
        base_thickness = 3 + group_idx
        
        print(f"   Group {group_idx+1} ({color}): {meas_name}: {value:.1f}{unit}")
        
        # FIRST line gets TEXT, others get thicker lines only
        for i in range(len(group_keypoints) - 1):
            kp1 = group_keypoints[i]
            kp2 = group_keypoints[i + 1]
            text = f"{meas_name}: {value:.1f}{unit}"
            
            if i == 0:  #FIRST line of group = TEXT
                img = draw_line_with_text(img, kp1, kp2, text, color, base_thickness)
            else:       #SUBSEQUENT lines = THICKER, NO TEXT
                thickness = base_thickness + 2
                img = draw_line_only(img, kp1, kp2, color, thickness)
    
    #header
    unit = measurements[0]['unit'] if measurements else 'cm'
    ratio = state_data['calib_result']['ratio']
    header = f"📏 {len(measurements)} groups | {ratio:.6f}px/{unit}"
    cv2.putText(img, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the Result
    result_img_path = result_folder / "result.jpg"
    cv2.imwrite(str(result_img_path), img)
    
    print(f"SAVED: {result_img_path}")
    print("RED points +  1 text/group + thicker overlaps")
    
    # cv2.imshow('📏 Clean Groups', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return str(result_img_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_from_result.py <result.json>")
        sys.exit(1)
    visualize_result(sys.argv[1])
