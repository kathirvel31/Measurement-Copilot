import cv2
from PIL import Image, ImageFilter
import numpy as np
import os
from datetime import datetime  
from config import config
def drawCalibrationText(result, pixel_to_cm_ratio, timestamp, i):
    """Draw calibration text - COLORS PRESERVED!"""
    img_cv = np.array(result).copy() 
    h, w = img_cv.shape[:2]
    
    # 2. Center text
    center_x, center_y = w // 2, h // 2
    text = f"Scale: {pixel_to_cm_ratio:.4f} cm/px"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    text_x = center_x - text_width // 2
    text_y = center_y
    
    # 3. Black BG rect (RGBA: B=0,G=0,R=0,A=255)
    cv2.rectangle(img_cv, 
                 (text_x-10, text_y-text_height-8), 
                 (text_x+text_width+10, text_y+8), 
                 (0, 0, 0, 255), -1)  # BGRA black + opaque
    
    # 4. White text (RGBA: B=255,G=255,R=255,A=255)
    cv2.putText(img_cv, text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 2)
    
    # 5. Back to PIL RGBA - perfect match!
    img_with_txt = Image.fromarray(img_cv, mode="RGBA")
    
    os.makedirs(f"runs/crop_pil/{timestamp}", exist_ok=True)
    save_path = f"runs/crop_pil/{timestamp}/dress_cross{i}.png"
    img_with_txt.save(save_path, "PNG")
    
    return {
        "img_array": img_with_txt, 
        "img_path": save_path,
        "scale_used": pixel_to_cm_ratio,
        "center_cm": (center_x * pixel_to_cm_ratio, center_y * pixel_to_cm_ratio)
    }



def crop_pil(image_path, foreground_masks_np):
    isShowCalibrationText = config.get('preferences.isShowCalibrationText')
    pixel_to_cm_ratio = config.get('calibration.pixel_to_cm_ratio')
    crop_result = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for i, foreground_mask in enumerate(foreground_masks_np):
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        if foreground_mask.max() <= 1.0: 
            alpha = (foreground_mask * 255).astype(np.uint8)
        else:
            alpha = foreground_mask.astype(np.uint8)
        img_array[:, :, 3] = alpha 
        result = Image.fromarray(img_array)

        
        if isShowCalibrationText:
            img_with_txt_result = drawCalibrationText(result, pixel_to_cm_ratio, timestamp, i)
            crop_result.append(img_with_txt_result)
        else:
            os.makedirs(f"runs/crop_pil/{timestamp}", exist_ok=True)
            result.save(f"runs/crop_pil/{timestamp}/dress_masked{i}.png", "PNG")
            crop_result.append({"img_array": result, "img_path": f"runs/crop_pil/{timestamp}/dress_masked{i}.png"})
    return crop_result