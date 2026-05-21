from ultralytics import YOLO
import cv2
import numpy as np
from config import config


def get_best_detection(results):
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        confs = result.boxes.conf.cpu().numpy()
        print("Confidences:", confs)
        best_idx = int(np.argmax(confs))
        result.boxes = result.boxes[best_idx:best_idx+1]
    return results


def detect_dress(source_img, output_dir="runs/detect"):
    model_path = config.get('models.det_dress')
    model = YOLO(model_path)
    img = cv2.imread(source_img)
    if img is None:
        print(f" Cannot load image: {source_img}")
        return
    results = model.predict(
        source=source_img,
        conf=0.1,  
    )
    best_results = get_best_detection(results)
    return best_results