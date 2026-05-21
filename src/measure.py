
"""      measurements - - FULLY UPDATED for new config.yaml (cm) + complex offsets
        Converting Pixels, units-- cm, mm 

 """
import json
import math
from src.state import state  
from config import config
from typing import Dict, List, Any, Optional

def load_model_constants(file_path: str = 'model_constants.json') -> Dict:
    """Load measurement rules from model_constants.json."""
    print(" Loading model_constants.json...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(" Model constants loaded successfully")
    return data

def convert_offset_to_pixels(offset_value: float, offset_unit: str, ratio: float, config_data: Dict) -> float:
    """Convert offset → pixels using NEW nested config structure."""
    main_unit = config_data.get('calibration.unit')  # Now "cm"
    print(f" Converting offset: {offset_value} {offset_unit} → pixels (main_unit={main_unit})")
    
    conversions = config_data.get('calibration.unit_conversions', {})
    
    #  unit_conversions[main_unit][target_unit]
    if offset_unit != main_unit and offset_unit in conversions.get(main_unit, {}):
        # Convert offset_unit → main_unit (cm)
        conversion_factor = conversions[offset_unit][main_unit]
        offset_main_unit = offset_value * conversion_factor
        print(f"  → {offset_main_unit:.1f} {main_unit} (×{conversion_factor})")
    else:
        offset_main_unit = offset_value
        print(f"  → {offset_main_unit:.1f} {main_unit} (no conversion)")
    
    # Convert main unit → pixels
    offset_pixels = offset_main_unit / ratio
    print(f"  → {offset_pixels:.1f} pixels (÷ {ratio:.4f})")
    
    return offset_pixels

def resolve_keypoint(kp_ref: Any, keypoints: List[List[float]], ratio: float, config_data: Dict) -> Optional[List[float]]:
    """Return RAW PIXEL coordinates."""
    if isinstance(kp_ref, int):
        if 0 <= kp_ref < len(keypoints):
            x, y = keypoints[kp_ref]
            print(f" Direct kp[{kp_ref}]: ({x:.1f}, {y:.1f})px")
            return [x, y]
        print(f" Keypoint index {kp_ref} out of range")
        return None
    
    if isinstance(kp_ref, dict):
        base_idx = kp_ref['keypoint']
        if 0 <= base_idx < len(keypoints):
            x, y = keypoints[base_idx]
            print(f" Base kp[{base_idx}]: ({x:.1f}, {y:.1f})px")
            offset_pixels = convert_offset_to_pixels(
                kp_ref['value'], 
                kp_ref.get('unit', 'cm'),  # Default to main unit
                ratio, 
                config_data
            )
            
            move = kp_ref.get('move', '')
            if move == 'up': 
                y -= offset_pixels
                print(f"  ↑ up {offset_pixels:.1f}px")
            elif move == 'down': 
                y += offset_pixels
                print(f"  ↓ down {offset_pixels:.1f}px")
            elif move == 'left': 
                x -= offset_pixels
                print(f"  ← left {offset_pixels:.1f}px")
            elif move == 'right': 
                x += offset_pixels
                print(f"  → right {offset_pixels:.1f}px")
            
            print(f" Final offset pt: ({x:.1f}, {y:.1f})px")
            return [x, y]
        print(f" Base keypoint {base_idx} out of range")
    return None

def euclidean_distance_px(p1: List[float], p2: List[float]) -> float:
    """Distance between two points in PIXELS."""
    if not p1 or not p2:
        return 0.0
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def handle_dist_keypoints(keypoints_list: List[Any], keypoints: List[List[float]], 
                         ratio: float, config_data: Dict) -> List[List[float]]:
    """Resolve all keypoints to pixel coordinates."""
    print(" Resolving keypoints list...")
    resolved = []
    for i, kp in enumerate(keypoints_list):
        point = resolve_keypoint(kp, keypoints, ratio, config_data)
        if point:
            resolved.append(point)
            print(f"  {i}: ✅ Resolved ({point[0]:.1f}, {point[1]:.1f})px")
        else:
            print(f"  {i}: ❌ Failed")
    print(f" Resolved {len(resolved)}/{len(keypoints_list)} points")
    return resolved

def calculate_measurement(keypoints_list: List[Any], keypoints: List[List[float]], 
                         ratio: float, config_data: Dict) -> float:
    """Calculate PIXEL distance first, then convert total to cm."""
    main_unit = config_data.get('calibration.unit')  # Now "cm"
    print(f" Calculating measurement with {len(keypoints_list)} keypoints...")
    
    if len(keypoints_list) < 2:
        print(" Less than 2 keypoints")
        return 0.0
    
    resolved_points_px = handle_dist_keypoints(keypoints_list, keypoints, ratio, config_data)
    valid_points = [p for p in resolved_points_px if p is not None]
    
    if len(valid_points) < 2:
        print("Less than 2 valid points")
        return 0.0
    
    print(f"Chaining {len(valid_points)} valid points...")
    total_pixels = 0.0
    for i in range(len(valid_points) - 1):
        dist_px = euclidean_distance_px(valid_points[i], valid_points[i+1])
        dx = valid_points[i+1][0] - valid_points[i][0]
        dy = valid_points[i+1][1] - valid_points[i][1]
        print(f"  Segment {i+1}: Δx={dx:.1f}px, Δy={dy:.1f}px → {dist_px:.1f}px")
        total_pixels += dist_px
    
    #  TOTAL pixel distance to main unit (cm) Convertions
    total_units = total_pixels * ratio
    print(f"Total: {total_pixels:.1f}px * {ratio:.6f} = {total_units:.2f}{main_unit}")
    return [round(total_units, 2), valid_points]

def validate_keypoints_match(state_detection: Dict, model_detection: Dict) -> bool:
    """Check if keypoints_count matches keypoints_num."""
    state_count = state_detection.get('keypoints_count')
    model_count = model_detection.get('keypoints_num')
    
    print(f" Validating keypoints: state={state_count}, model={model_count}")
    if state_count == model_count:
        print(" Keypoints count MATCH")
        return True
    print(f" Keypoints MISMATCH {state_count} != {model_count}")
    return False

def compute_all_measurements(constants_file: str = 'model_constants.json') -> List[Dict]:
    """Main function: Use state.py + compute measurements in cm."""
    print(" Starting measurements computation...")
    
    current_state = state.state
    constants = load_model_constants(constants_file)
    
    ratio = current_state['calib_result']['ratio']  #  eg: 0.04648 cm/px
    main_unit = config.get('calibration.unit') or 'cm'      # "cm"
    
    if not ratio:
        print(" valid calibration ratio not found")
        ratio = config.get('calibration.pixel_to_unit_ratio')
        if not ratio:
             print(" No calibration ratio available in config or state. Using 1.0 px/unit as fallback (MEASUREMENTS WILL BE WRONG)")
             ratio = 1.0
        else:
             print(f" Using ratio from config: {ratio}")

    print(f"Calibration ratio: 1px = {ratio} {main_unit}")
    print(f" Main unit: {main_unit}")
    
    measurements = []
    
    if 'keypoints_result' not in current_state or 'detections' not in current_state['keypoints_result']:
        print(" No keypoint detections found in state.")
        return []

    keypoints_detections = current_state['keypoints_result']['detections']
    
    print(f" Found {len(keypoints_detections)} detections in state")
    
    for det_idx, state_det in enumerate(keypoints_detections):
        print(f"\n{'='*60}")
        print(f"Processing detection {det_idx+1}: class_id={state_det['class_id']}")
        print(f"   Class: {state_det.get('class_name', 'unknown')}")
        
        state_class_id = state_det['class_id']
        
        model_det = None
        for const_det in constants.get('keypints_model', {}).get('detections', []):
            if const_det.get('class_id') == state_class_id:
                model_det = const_det
                break
        
        if not model_det:
            print(f"No model constants for class_id {state_class_id}")
            continue
        
        print(f"Found matching model: {model_det.get('class_name', 'unknown')}")
        
        if not validate_keypoints_match(state_det, model_det):
            continue
        
        keypoints = state_det['keypoints']
        class_name = state_det.get('class_name', 'unknown')
        measurement_pairs = model_det.get('measurement_pairs', [])
        print(f" Processing {len(measurement_pairs)} measurement pairs...")
        
        for pair_idx, pair in enumerate(measurement_pairs):
            if 'keypoints' not in pair or len(pair['keypoints']) < 2:
                print(f"  {pair_idx+1}: Invalid keypoints")
                continue
            
            print(f"\n  📐 Measurement {pair_idx+1}: {pair.get('measurement_name', 'unnamed')}")
            kp_list = pair['keypoints']
            meas_name = pair.get('measurement_name', f"{class_name}_unnamed")
            
            print(f"     Keypoints: {kp_list}")
            
            value_pack = calculate_measurement(kp_list, keypoints, ratio, config)
            value = value_pack[0]
            valid_points = value_pack[1]
            
            if value > 0:
                measurement = {
                    'class_name': class_name,
                    'class_id': state_class_id,
                    'measurement_pair': meas_name,
                    'value': value,
                    'unit': main_unit,
                    'keypoints_classID': kp_list,
                    'keypoints': valid_points,
                    # 'raw_pixels': round(value / ratio, 1)
                }
                measurements.append(measurement)
                print(f"SAVED: {meas_name} = {value:.2f} {main_unit}")
            else:
                print(f"ZERO measurement skipped")
    
    # Auto-save to state
    if measurements:
        state.update_measuremnts(measurements)

    print(f"\n COMPLETED: {len(measurements)} measurements computed!")
    return measurements
