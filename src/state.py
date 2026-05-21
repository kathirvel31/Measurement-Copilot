
# state.py
import json
import os
from datetime import datetime
from pathlib import Path
from config import config
from src.visualize_result import visualize_result

class StateManager:
    def __init__(self, state_file="state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()
    
    def _get_default_state(self):
        """Default empty state matching your exact structure"""
        return {
            "date_time": "",
            "img_path": "",
            "calib_result": {
                "ratio": 0.0,
                "ref_obj": "",
                "ref_ob_id": 0
            },
            "det_result": [],
            "keypoints_result": {},
            "measurements_result": []
        }
    
    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                print(" Invalid state.json, using default")
        return self._get_default_state()
    
    def save(self):
        os.makedirs(self.state_file.parent, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f" State saved to {self.state_file}")
    
    def save_final_state(self):
        if config.get('preferences.saveState'):
            timestamp = self.state.get('date_time', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            final_dir = Path(f"runs/results/{timestamp}")
            final_dir.mkdir(parents=True,exist_ok=True)
            final_path = final_dir / f"result.json"
            with open(final_path, 'w') as f:
                json.dump(self.state, f, indent=2)
            if config.get('preferences.visualizeResult'):
                visualize_result(str(final_path))
            print(f" Final state saved: {final_path}")
    
    def reset_state(self):
        print(" Resetting state...")
        self.save_final_state()
        self.state = self._get_default_state()
        self.save()
        print(" State reset complete")
    
    def update_img(self, img_path):
        self.state["img_path"] = str(img_path)
        self.state["date_time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save()
    
    def update_calib(self, ratio, ref_obj, ref_id):
        self.state["calib_result"] = {
            "ratio": float(ratio),
            "ref_obj": str(ref_obj),
            "ref_ob_id": int(ref_id)
        }
        self.save()
    
    def update_det_result(self, det_output):
        """Update detection results - EXACTLY your format (NO keypoints ever!)"""
        self.state["img_path"] = det_output.get('img_path', self.state["img_path"])
        
        detections = det_output.get('detections', [])
        self.state["det_result"] = detections  # Store EXACTLY as provided
        
        self.save()
        print(f" Updated {len(detections)} detections (NO keypoints)")
    
    def update_keypoints(self, kpt_output):
        """Store COMPLETE keypoint model output directly under keypoints_result ONLY"""
        try:
            if not kpt_output or 'detections' not in kpt_output:
                print(" Invalid keypoint output format")
                return
            
            # Store full keypoint output directly - SEPARATE from detection
            self.state["keypoints_result"] = kpt_output
            
            self.save()
            detections = kpt_output.get('detections', [])
            num_kpts = len(kpt_output['detections'][0]['keypoints']) if detections else 0
            print(f" Keypoints stored: {len(detections)} detections, {num_kpts} keypoints")
            
        except Exception as e:
            print(f" Keypoint update failed: {e}")
    
    def update_measuremnts(self, measurements_output):
        """Store COMPLETE keypoint model output directly under measurements_result ONLY"""
        try:
            self.state["measurements_result"] = measurements_output
            self.save()
            print(f" Measurements stored: {len(measurements_output)} measurements")
            
        except Exception as e:
            print(f" Keypoint update failed: {e}")
    
    def get_crop_info(self, det_id=0):
        """Get crop coordinates from det_result"""
        if det_id < len(self.state["det_result"]):
            bbox = self.state["det_result"][det_id]["bbox"]
            return {
                "x1": int(bbox["x1"]),
                "y1": int(bbox["y1"]),
                "x2": int(bbox["x2"]),
                "y2": int(bbox["y2"])
            }
        return None
    
    def print_state(self):
        print("\n Current State:")
        print(json.dumps(self.state, indent=2))

# Global instance
state = StateManager()
