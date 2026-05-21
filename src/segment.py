from ultralytics import SAM
from src.crop_pil import crop_pil
from config import config

def segment_dress(results, image_path):
    try:
        model_path = config.get('models.seg_dress')
        sam_model = SAM(model_path)
        result_imgs= []
        for result in results:
            classId = result.boxes.cls.int().tolist()
            if len(classId):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    x1, y1, x2, y2 = boxes[0]
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2 
                    sam_results = sam_model(
                        result.orig_img, 
                        bboxes=boxes, 
                        points=[[center_x, center_y]],  
                        labels=[1], 
                        verbose=False, 
                        save= True, 
                        conf=0.4 ,
                        device='cpu')
                    sam_res = sam_results[0]
                    masks_np = sam_res.masks.data.cpu().numpy()
                    masked_img = crop_pil(image_path, masks_np)
                    result_imgs.append({"img_path": image_path, "crop_imgs": masked_img})
        return result_imgs
    except Exception as e:
        print(f" Error during segmentation: {e}")
        return None