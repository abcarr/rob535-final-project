"""
Investigate NAVSIM metadata .pkl file structure

Usage:
    python investigate_metadata.py --scene 2021.05.12.19.36.12_veh-35_00005_00204
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from pprint import pprint


def investigate_metadata(scene_name, metadata_root):
    """Thoroughly investigate metadata file structure."""
    
    pkl_file = Path(metadata_root) / f"{scene_name}.pkl"
    
    if not pkl_file.exists():
        print(f"❌ Metadata not found: {pkl_file}")
        return
    
    print("=" * 80)
    print(f"Investigating Metadata: {scene_name}")
    print("=" * 80)
    
    # Load metadata
    data = pickle.load(open(pkl_file, 'rb'))
    
    print(f"\n📦 TOP LEVEL STRUCTURE")
    print(f"  Type: {type(data)}")
    print(f"  Length: {len(data)} frames")
    
    # Investigate first frame in detail
    frame = data[0]
    
    print(f"\n📋 FRAME 0 KEYS ({len(frame.keys())} total):")
    for key in sorted(frame.keys()):
        value = frame[key]
        value_type = type(value).__name__
        
        # Get size/shape info
        if isinstance(value, (list, tuple)):
            size_info = f"len={len(value)}"
        elif isinstance(value, np.ndarray):
            size_info = f"shape={value.shape}, dtype={value.dtype}"
        elif isinstance(value, dict):
            size_info = f"{len(value)} keys"
        elif isinstance(value, (int, float, str)):
            size_info = f"value={value}"
        else:
            size_info = ""
        
        print(f"  • {key:30s} {value_type:15s} {size_info}")
    
    # Deep dive into important fields
    print(f"\n" + "=" * 80)
    print("DETAILED FIELD INSPECTION")
    print("=" * 80)
    
    # 1. Camera info
    print(f"\n📷 CAMERAS ('cams' field):")
    cams = frame['cams']
    print(f"  Number of cameras: {len(cams)}")
    print(f"  Camera names: {list(cams.keys())}")
    
    # Pick CAM_F0 as example
    cam_f0 = cams['CAM_F0']
    print(f"\n  CAM_F0 structure ({len(cam_f0)} keys):")
    for key in sorted(cam_f0.keys()):
        value = cam_f0[key]
        if isinstance(value, np.ndarray):
            print(f"    • {key:30s} shape={value.shape}, dtype={value.dtype}")
            if value.size < 20:
                print(f"      {value}")
        else:
            print(f"    • {key:30s} {type(value).__name__} = {value}")
    
    # 2. LiDAR info
    print(f"\n🔭 LIDAR:")
    print(f"  lidar_path: {frame['lidar_path']}")
    
    # 3. Ego state info
    print(f"\n🚗 EGO STATE:")
    if 'ego2global_translation' in frame:
        print(f"  ego2global_translation: {frame['ego2global_translation']}")
    if 'ego2global_rotation' in frame:
        print(f"  ego2global_rotation: {frame['ego2global_rotation']}")
    if 'ego_dynamic_state' in frame:
        ego_dyn = frame['ego_dynamic_state']
        if isinstance(ego_dyn, np.ndarray):
            print(f"  ego_dynamic_state: shape={ego_dyn.shape}")
            print(f"    {ego_dyn}")
        else:
            print(f"  ego_dynamic_state: {ego_dyn}")
    
    # 4. Other sensors
    print(f"\n🔬 OTHER FIELDS:")
    interesting_keys = ['token', 'timestamp', 'roadblock_ids', 'driving_command', 
                        'annotations', 'agents', 'tracking']
    
    for key in interesting_keys:
        if key in frame:
            value = frame[key]
            if isinstance(value, np.ndarray):
                print(f"  • {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  • {key}: {type(value).__name__} with {len(value)} elements")
                if len(value) > 0 and len(value) <= 3:
                    print(f"      {value}")
            elif isinstance(value, dict):
                print(f"  • {key}: dict with {len(value)} keys: {list(value.keys())[:5]}...")
            else:
                print(f"  • {key}: {value}")
    
    # 5. Check multiple frames for temporal info
    print(f"\n⏱️  TEMPORAL STRUCTURE:")
    print(f"  Total frames in scene: {len(data)}")
    
    # Check if timestamps/paths change across frames
    if len(data) >= 3:
        print(f"\n  Frame 0 → 1 → 2 comparison:")
        for i in range(3):
            cam_path = data[i]['cams']['CAM_F0']['data_path']
            lidar_path = data[i]['lidar_path']
            
            cam_file = Path(cam_path).name
            lidar_file = Path(lidar_path).name
            
            print(f"    Frame {i}:")
            print(f"      Camera: {cam_file}")
            print(f"      LiDAR:  {lidar_file}")
            
            if 'timestamp' in data[i]:
                print(f"      Timestamp: {data[i]['timestamp']}")
    
    # 6. Camera-LiDAR pairing analysis
    print(f"\n🔗 CAMERA-LIDAR PAIRING:")
    print(f"  Checking first 5 frames...")
    
    for i in range(min(5, len(data))):
        cam_stem = Path(data[i]['cams']['CAM_F0']['data_path']).stem
        lidar_stem = Path(data[i]['lidar_path']).stem
        match = cam_stem == lidar_stem
        
        match_str = "✓" if match else "✗"
        print(f"  Frame {i}: {match_str} CAM={cam_stem[:16]}... LIDAR={lidar_stem[:16]}... Match={match}")
    
    # 7. Coordinate frame transformations
    print(f"\n🌐 COORDINATE TRANSFORMATIONS:")
    cam_f0 = frame['cams']['CAM_F0']
    
    if 'sensor2lidar_rotation' in cam_f0:
        R = cam_f0['sensor2lidar_rotation']
        t = cam_f0['sensor2lidar_translation']
        
        print(f"  sensor2lidar_rotation (camera → LiDAR):")
        print(f"    {R}")
        print(f"\n  sensor2lidar_translation (camera → LiDAR):")
        print(f"    {t}")
        
        # Verify it's a valid rotation matrix
        RtR = R.T @ R
        det = np.linalg.det(R)
        is_valid = np.allclose(RtR, np.eye(3)) and np.abs(det - 1.0) < 1e-5
        
        print(f"\n  Rotation matrix valid: {is_valid}")
        print(f"    R^T @ R ≈ I: {np.allclose(RtR, np.eye(3))}")
        print(f"    det(R) ≈ 1: {np.abs(det - 1.0) < 1e-5} (det={det:.6f})")
    
    # 8. Check if depth/semantic ground truth exists
    print(f"\n🎯 GROUND TRUTH LABELS:")
    gt_fields = ['depth_gt', 'semantic_gt', 'instance_gt', 'occupancy_gt', 
                 'depth_map', 'segmentation', 'labels']
    
    found_gt = []
    for field in gt_fields:
        if field in frame:
            found_gt.append(field)
            value = frame[field]
            if isinstance(value, np.ndarray):
                print(f"  ✓ {field}: shape={value.shape}")
            else:
                print(f"  ✓ {field}: {type(value).__name__}")
    
    if not found_gt:
        print(f"  ✗ No direct depth/semantic ground truth found")
        print(f"  → Will need to project LiDAR or use pseudo-labels")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✓ Scene has {len(data)} frames")
    print(f"✓ Each frame has {len(frame.keys())} fields")
    print(f"✓ Cameras: {len(cams)} views")
    print(f"✓ Camera intrinsics: {cam_f0['cam_intrinsic'].shape}")
    print(f"✓ Camera → LiDAR transform: Available")
    
    # Check timestamp matching
    matches = sum(1 for i in range(min(5, len(data))) 
                  if Path(data[i]['cams']['CAM_F0']['data_path']).stem == 
                     Path(data[i]['lidar_path']).stem)
    
    if matches > 0:
        print(f"✓ Camera-LiDAR timestamps: MATCH ({matches}/5 tested)")
    else:
        print(f"✗ Camera-LiDAR timestamps: DON'T MATCH")
        print(f"  → Metadata pairs them correctly, don't match by filename!")
    
    if found_gt:
        print(f"✓ Ground truth fields: {', '.join(found_gt)}")
    else:
        print(f"✗ No direct 2D ground truth (will project from 3D)")


def main():
    parser = argparse.ArgumentParser(description="Investigate NAVSIM metadata structure")
    parser.add_argument(
        "--scene",
        type=str,
        default="2021.05.12.19.36.12_veh-35_00005_00204",
        help="Scene name to investigate"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="navsim_logs/trainval",
        help="Path to metadata directory"
    )
    
    args = parser.parse_args()
    investigate_metadata(args.scene, args.metadata)


if __name__ == "__main__":
    main()
