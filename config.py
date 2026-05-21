
import yaml
import os
from datetime import datetime

CONFIG_FILE = "config.yaml"

class Config:
    def __init__(self):
        self.data = self.load_config()
    
    def load_config(self):
        """Load config.yaml"""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return yaml.safe_load(f)
        else:
            print("❌ config.yaml not found! Creating default...")
            return self.create_default_config()
    
    def create_default_config(self):
        """Create default config if missing"""
        default = {
            'calibration': {
                'pixel_to_cm_ratio': None,
                'calibrated_at': None,
                'unit': 'm',
                'unit_conversions':{
                    'mm': {'m': 0.001, 'cm': 0.1, 'mm': 1},
                    'cm': {'m': 0.01, 'cm': 1, 'mm': 10},
                    'm': {'m': 1, 'cm': 100, 'mm': 1000},
                }
            },
            'preferences': {
                'optimizeCalibration': False,
                'handleMultipleDresses': False,
                'saveState': True,
                'visualizeResult': True
            },
            'debug':{
                'showCalibrationText': False,
            },  
            'models':{
                'det_coin':'models/coin_det.pt', # Placeholder
                'seg_dress':'models/sam2_l.pt',
                'det_dress':'models/yolo11n-dress.pt',
                'det_dress_keypoints': 'models/yolov11-pose1000.pt'
            },
            'defaults': {
                'pixel_to_cm_ratio': None,
                'test_img': 'data/test_img.jpg', # Placeholder
            },
            'conf': 0.25,
            'iou': 0.7
        }
        self.save_config(default)
        return default
    
    def save_config(self, data=None):
        """Save config.yaml"""
        if data is None:
            data = self.data
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get(self, key_path, default=None):
        """Get nested value: config.get('calibration.pixel_to_cm_ratio')"""
        keys = key_path.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key_path, value):
        keys = key_path.split('.')
        current = self.data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        self.save_config()
        print(f"✅ Set {key_path} = {value}")

    
    def update_calibration(self, pixel_to_cm_ratio):
        """Update calibration from coin detection"""
        if pixel_to_cm_ratio is not None:
            self.set('calibration.pixel_to_cm_ratio', float(pixel_to_cm_ratio))
        else:
            self.set('calibration.pixel_to_cm_ratio', None)
        self.set('calibration.calibrated_at', datetime.now().isoformat())

        if pixel_to_cm_ratio is None:
            print("⚠️ Calibration updated: pixel_to_cm_ratio is None")
        else:
            print(f"✅ Calibration updated: {pixel_to_cm_ratio:.6f} {self.get('calibration.unit')}/px")

# Global config instance
config = Config()
