
"""This Function defines the Keypoints Detection From the Garment Image 
    Keypoints --->> State.json
 """

from ultralytics import YOLO
import cv2
import numpy as np
from config import config
# Import state from src.state since we are in src directory or calling from root
from src.state import state


def process_data(results):
    detections = []
    for r in results:
        # Loop through each detection in this image
        num_detections = len(r.boxes) if r.boxes is not None else 0
        
        for i in range(num_detections):
            detection = {}
            
            # 1. Get class ID and name
            cls_id = int(r.boxes.cls[i])  
            class_name = r.names[cls_id]  #shirt
            detection['class_id'] = cls_id
            detection['class_name'] = class_name
            
            # 2. Get bounding box
            box = r.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
            detection['bbox'] = {
                'x1': float(box[0]),
                'y1': float(box[1]), 
                'x2': float(box[2]),
                'y2': float(box[3])
            }
            
            # 3. Get confidence
            detection['confidence'] = float(r.boxes.conf[i])
            
            # 4. Get keypoints (if available)
            if r.keypoints is not None and i < len(r.keypoints):
                keypoints = r.keypoints.xy[i].tolist()  # [[x1,y1], [x2,y2], ...]
                detection['keypoints_count']= len(keypoints)
                detection['keypoints'] = keypoints  # List of [x,y] points
                
                # Also get visibility/confidence per keypoint if available
                if r.keypoints.has_visible:
                    vis = r.keypoints.data[i][2::3].tolist()  # Visibility scores
                    detection['keypoint_visibility'] = vis
            
            detections.append(detection)
    return detections

def detect_keypoints(source_img, output_dir="runs/keypoints", save_img=True):
    try:
        model_path = config.get('models.det_dress_keypoints')
        if not model_path:
             print(" Keypoints model path not found in config!")
             return None
        model = YOLO(model_path)
        
        # Get thresholds from config
        conf_threshold = config.get('conf', 0.25)        # Box confidence
        # kpt_conf_threshold = config.get('kpt_conf', 0.3) # Keypoint confidence
        
        results = model(
            source_img, 
            save=save_img, 
            project=output_dir,
            conf=conf_threshold,           # Box detection threshold
            iou=config.get('iou', 0.7),    # NMS IoU
            verbose=False
        )

        if results is not None and len(results) != 0:
            detections = process_data(results)
            state.update_keypoints({
                'img_path': source_img,
                'detections': detections
            })
            return {"img_path": source_img, "detections": detections}
        else:
            return None
    except Exception as e:
        print(f"Error during keypoint prediction: {e}")
        return None
