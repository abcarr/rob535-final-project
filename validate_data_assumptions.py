#!/usr/bin/env python3
"""
Quick validation of data structure assumptions before running full multitask test.
This checks the format/shape of metadata fields we need.
"""

import numpy as np
import pickle
from pathlib import Path

# Paths
METADATA_FILE = Path("navsim_logs/trainval/2021.05.12.19.36.12_veh-35_00005_00204/scene_metadata.pkl")
FRAME_IDX = 0

print("=" * 80)
print("DATA STRUCTURE VALIDATION")
print("=" * 80)

# Load metadata
if not METADATA_FILE.exists():
    print(f"❌ Metadata file not found: {METADATA_FILE}")
    print("\nPlease update paths to match your setup:")
    print("   - METADATA_FILE: path to scene_metadata.pkl")
    exit(1)

with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

print(f"\n✅ Loaded metadata with {len(metadata)} frames\n")

frame = metadata[FRAME_IDX]

# ============================================================================
# 1. Camera Data Structure
# ============================================================================
print("=" * 80)
print("1. CAMERA DATA (CAM_F0)")
print("=" * 80)

cam_f0 = frame['cams']['CAM_F0']
print(f"\nAvailable keys: {cam_f0.keys()}")

# Intrinsics
intrinsics = np.array(cam_f0['intrinsics'])
print(f"\nIntrinsics:")
print(f"  - Type: {type(cam_f0['intrinsics'])}")
print(f"  - Shape: {intrinsics.shape}")
print(f"  - Values:\n{intrinsics}")

# Extrinsics
s2l_rotation = np.array(cam_f0['sensor2lidar_rotation'])
s2l_translation = np.array(cam_f0['sensor2lidar_translation'])
print(f"\nsensor2lidar_rotation:")
print(f"  - Shape: {s2l_rotation.shape}")
print(f"  - Values:\n{s2l_rotation}")
print(f"\nsensor2lidar_translation:")
print(f"  - Shape: {s2l_translation.shape}")
print(f"  - Values: {s2l_translation}")

# Verify rotation matrix properties
det = np.linalg.det(s2l_rotation)
is_orthogonal = np.allclose(s2l_rotation @ s2l_rotation.T, np.eye(3))
print(f"\n✓ Rotation matrix validation:")
print(f"  - Determinant: {det:.6f} (should be ±1)")
print(f"  - Orthogonal: {is_orthogonal} (should be True)")

# ============================================================================
# 2. Annotations (3D Boxes)
# ============================================================================
print("\n" + "=" * 80)
print("2. ANNOTATIONS (3D BOUNDING BOXES)")
print("=" * 80)

anns = frame['anns']
print(f"\nAvailable keys: {anns.keys()}")

gt_boxes = anns['gt_boxes']
gt_names = anns['gt_names']
instance_tokens = anns['instance_tokens']

print(f"\ngt_boxes:")
print(f"  - Type: {type(gt_boxes)}")
print(f"  - Shape: {gt_boxes.shape}")
print(f"  - First box: {gt_boxes[0]}")
print(f"  - Expected format: [x, y, z, width, length, height, yaw]")

print(f"\ngt_names:")
print(f"  - Type: {type(gt_names)}")
print(f"  - Shape: {gt_names.shape}")
print(f"  - First name: {gt_names[0]}")
print(f"  - Unique classes: {np.unique(gt_names)}")

print(f"\ninstance_tokens:")
print(f"  - Type: {type(instance_tokens)}")
print(f"  - Shape: {instance_tokens.shape}")
print(f"  - First token: {instance_tokens[0]}")

# Check box value ranges
print(f"\nBox statistics:")
print(f"  - X range: [{gt_boxes[:, 0].min():.1f}, {gt_boxes[:, 0].max():.1f}]")
print(f"  - Y range: [{gt_boxes[:, 1].min():.1f}, {gt_boxes[:, 1].max():.1f}]")
print(f"  - Z range: [{gt_boxes[:, 2].min():.1f}, {gt_boxes[:, 2].max():.1f}]")
print(f"  - Width range: [{gt_boxes[:, 3].min():.1f}, {gt_boxes[:, 3].max():.1f}]")
print(f"  - Length range: [{gt_boxes[:, 4].min():.1f}, {gt_boxes[:, 4].max():.1f}]")
print(f"  - Height range: [{gt_boxes[:, 5].min():.1f}, {gt_boxes[:, 5].max():.1f}]")
print(f"  - Yaw range: [{gt_boxes[:, 6].min():.2f}, {gt_boxes[:, 6].max():.2f}] rad")

# ============================================================================
# 3. LiDAR Data
# ============================================================================
print("\n" + "=" * 80)
print("3. LIDAR DATA")
print("=" * 80)

lidar_path = frame['lidar_path']
print(f"\nLiDAR path: {lidar_path}")

# Try to load LiDAR
scene_dir = Path("sensor_blobs/trainval/2021.05.12.19.36.12_veh-35_00005_00204")
if (scene_dir / lidar_path).exists():
    lidar_pc = np.fromfile(scene_dir / lidar_path, dtype=np.float32).reshape(-1, 5)
    print(f"\n✅ Loaded LiDAR:")
    print(f"  - Shape: {lidar_pc.shape}")
    print(f"  - Fields: [x, y, z, intensity, ring] (assumed)")
    print(f"  - XYZ range:")
    print(f"    - X: [{lidar_pc[:, 0].min():.1f}, {lidar_pc[:, 0].max():.1f}]")
    print(f"    - Y: [{lidar_pc[:, 1].min():.1f}, {lidar_pc[:, 1].max():.1f}]")
    print(f"    - Z: [{lidar_pc[:, 2].min():.1f}, {lidar_pc[:, 2].max():.1f}]")
else:
    print(f"⚠️  LiDAR file not found at: {scene_dir / lidar_path}")

# ============================================================================
# 4. WoTE Image Processing
# ============================================================================
print("\n" + "=" * 80)
print("4. WoTE IMAGE DIMENSIONS")
print("=" * 80)

print("\nFrom WoTE_features.py analysis:")
print("  - Original camera: 1920×1080 (nuScenes)")
print("  - After crop: Remove 28px top/bottom → 1920×1024")
print("  - L0/R0 crop: 416px each side → 1088×1024")
print("  - Stitched: L0(1088) + F0(1920) + R0(1088) = 4096×1024")
print("  - Resized: cv2.resize to 1024×256")
print("\n✓ Target image shape: (256, 1024)")

# ============================================================================
# 5. Coordinate Frame Check
# ============================================================================
print("\n" + "=" * 80)
print("5. COORDINATE FRAME VERIFICATION")
print("=" * 80)

print("\n📍 Expected coordinate system (nuScenes/NAVSIM):")
print("  - Ego/LiDAR frame:")
print("    - X: forward (front of car)")
print("    - Y: left (left side of car)")
print("    - Z: up (top of car)")
print("  - Camera frame (sensor):")
print("    - X: right")
print("    - Y: down")
print("    - Z: forward")

print("\n📦 3D boxes should be in ego/LiDAR frame (based on WoTE code)")
print("  - Boxes around ego vehicle: X ≈ 0-50m, Y ≈ ±30m")
print("  - sensor2lidar transform converts camera → LiDAR frame")

# Quick sanity check
num_front = ((gt_boxes[:, 0] > 0) & (gt_boxes[:, 0] < 60)).sum()
num_behind = ((gt_boxes[:, 0] < 0) & (gt_boxes[:, 0] > -60)).sum()
print(f"\n✓ Sanity check:")
print(f"  - Objects in front (X > 0): {num_front}/{len(gt_boxes)}")
print(f"  - Objects behind (X < 0): {num_behind}/{len(gt_boxes)}")
print(f"  - Expected: More objects in front (matches camera FOV)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

checks = {
    "Camera intrinsics is 3×3": intrinsics.shape == (3, 3),
    "sensor2lidar_rotation is 3×3": s2l_rotation.shape == (3, 3),
    "sensor2lidar_translation is 3D": s2l_translation.shape == (3,),
    "Rotation matrix is valid": is_orthogonal and abs(abs(det) - 1.0) < 0.01,
    "gt_boxes has 7 values per box": gt_boxes.shape[1] == 7,
    "More objects in front": num_front > num_behind,
}

print("\n✅ Validation Results:")
all_passed = True
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")
    if not passed:
        all_passed = False

if all_passed:
    print("\n🎉 ALL CHECKS PASSED!")
    print("   → Safe to run test_multitask_target_generation.py")
else:
    print("\n⚠️  SOME CHECKS FAILED!")
    print("   → Review assumptions before running full test")

print("\n" + "=" * 80)
