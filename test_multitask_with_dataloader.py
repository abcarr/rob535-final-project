#!/usr/bin/env python3
"""
Test multitask target generation using NAVSIM's proper dataloader.
This ensures camera and LiDAR are correctly paired.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Use NAVSIM's dataloader to ensure proper camera/LiDAR pairing
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig, SceneFilter

# ============================================================================
# Configuration
# ============================================================================
# Scene token is the frame token, not the log filename!
# Use the first available scene from the log
SCENE_TOKEN = "a6a284584b19599f"  # From frame 3 of first log
FRAME_IDX = 3  # Middle frame (has history context)

# Image dimensions (front camera after WoTE preprocessing)
IMG_HEIGHT = 256
IMG_WIDTH = 1024

# Class mapping
CLASS_MAP = {
    'vehicle': 1,
    'pedestrian': 2,
    'bicycle': 3,
    'traffic_cone': 4,
    'barrier': 5,
    'construction': 6,
    'generic_object': 7,
}

# ============================================================================
# Helper Functions (same as before)
# ============================================================================

def get_box_corners_3d(box):
    """Get 8 corners of a 3D bounding box."""
    x, y, z, w, l, h, yaw = box
    
    corners_local = np.array([
        [-l/2, -w/2, -h/2], [-l/2, -w/2,  h/2],
        [-l/2,  w/2, -h/2], [-l/2,  w/2,  h/2],
        [ l/2, -w/2, -h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2, -h/2], [ l/2,  w/2,  h/2],
    ])
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
    
    corners_world = corners_local @ R.T + np.array([x, y, z])
    return corners_world

def project_3d_to_2d(points_3d, intrinsics, R, t, debug=False):
    """Project 3D points in ego/LiDAR frame to 2D camera image."""
    if debug:
        print(f"\n🔍 Projection Debug:")
        print(f"   Input (LiDAR frame): X=[{points_3d[:, 0].min():.1f}, {points_3d[:, 0].max():.1f}]")
        print(f"                        Y=[{points_3d[:, 1].min():.1f}, {points_3d[:, 1].max():.1f}]")
        print(f"                        Z=[{points_3d[:, 2].min():.1f}, {points_3d[:, 2].max():.1f}]")
    
    # Transform: p_camera = R^T @ (p_lidar - t)
    R_inv = R.T
    points_cam = (R_inv @ (points_3d - t).T).T
    
    if debug:
        print(f"   Camera frame: X=[{points_cam[:, 0].min():.1f}, {points_cam[:, 0].max():.1f}]")
        print(f"                 Y=[{points_cam[:, 1].min():.1f}, {points_cam[:, 1].max():.1f}]")
        print(f"                 Z=[{points_cam[:, 2].min():.1f}, {points_cam[:, 2].max():.1f}]")
        print(f"   Points in front (Z>0): {(points_cam[:, 2] > 0).sum()}/{len(points_cam)}")
    
    # Project to image
    points_2d_homogeneous = (intrinsics @ points_cam.T).T
    depths = points_2d_homogeneous[:, 2]
    
    valid_depth = depths > 0.1
    points_2d = np.zeros((len(points_3d), 2))
    points_2d[valid_depth] = points_2d_homogeneous[valid_depth, :2] / depths[valid_depth, None]
    
    u, v = points_2d[:, 0], points_2d[:, 1]
    valid_mask = valid_depth & (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
    
    return points_2d.astype(np.int32), depths, valid_mask

def generate_semantic_mask_from_boxes(boxes, names, intrinsics, R, t, img_shape):
    """Generate semantic segmentation mask from 3D boxes."""
    H, W = img_shape
    semantic_mask = np.zeros((H, W), dtype=np.int32)
    
    distances = np.linalg.norm(boxes[:, :2], axis=1)
    sorted_indices = np.argsort(-distances)
    
    num_visible = 0
    for idx in sorted_indices:
        box = boxes[idx]
        name = names[idx]
        
        semantic_class = CLASS_MAP.get(name, 0)
        if semantic_class == 0:
            continue
        
        corners_3d = get_box_corners_3d(box)
        corners_2d, depths, valid_mask = project_3d_to_2d(corners_3d, intrinsics, R, t)
        
        if valid_mask.sum() < 3:
            continue
        
        valid_corners = corners_2d[valid_mask]
        cv2.fillPoly(semantic_mask, [valid_corners], int(semantic_class))
        num_visible += 1
    
    coverage = (semantic_mask > 0).sum() / (H * W) * 100
    return semantic_mask, coverage, num_visible

def generate_depth_from_lidar(lidar_pc, intrinsics, R, t, img_shape):
    """Generate depth map from LiDAR projection."""
    H, W = img_shape
    depth_map = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)
    
    points_2d, depths, point_valid = project_3d_to_2d(lidar_pc, intrinsics, R, t)
    
    valid_points = points_2d[point_valid]
    valid_depths = depths[point_valid]
    
    for (u, v), d in zip(valid_points, valid_depths):
        if 0 <= v < H and 0 <= u < W:
            if not valid_mask[v, u] or d < depth_map[v, u]:
                depth_map[v, u] = d
                valid_mask[v, u] = True
    
    coverage = valid_mask.sum() / (H * W) * 100
    return depth_map, valid_mask, coverage

# ============================================================================
# Main Test
# ============================================================================

def main():
    print("=" * 80)
    print("MULTITASK TEST USING NAVSIM DATALOADER")
    print("=" * 80)
    
    # Initialize dataloader
    sensor_config = SensorConfig.build_all_sensors()
    
    # Create SceneFilter - required by SceneLoader
    # Load from the first log file only (fast for testing)
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        has_route=True,
        log_names=["2021.05.12.19.36.12_veh-35_00005_00204"],  # First log (filename without .pkl)
        max_scenes=5,  # Just load first 5 scenes
    )
    
    loader = SceneLoader(
        sensor_blobs_path=Path("sensor_blobs/trainval"),
        data_path=Path("navsim_logs/trainval"),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    
    # Load agent input (doesn't require map API like Scene does)
    print(f"\n📦 Loading scene: {SCENE_TOKEN}")
    agent_input = loader.get_agent_input_from_token(SCENE_TOKEN, use_fut_frames=False)
    print(f"   Ego statuses: {len(agent_input.ego_statuses)}")
    
    # Get current frame data (history frame 3 = current)
    current_idx = 3  # num_history_frames - 1
    print(f"\n📷 Using current frame (history index {current_idx})")
    
    # Get camera (front camera)
    camera = agent_input.cameras.cam_f0[current_idx]
    print(f"   Camera image shape: {camera.image.shape}")
    print(f"   Camera intrinsics: {camera.intrinsics.shape}")
    
    # Get LiDAR
    lidar = agent_input.lidars[current_idx]
    lidar_pc = lidar.lidar_pc[:3, :].T  # [N, 3]
    print(f"   LiDAR points: {len(lidar_pc)}")
    print(f"   LiDAR range: X=[{lidar_pc[:, 0].min():.1f}, {lidar_pc[:, 0].max():.1f}]m")
    
    # Get annotations from scene_frames_dict directly
    scene_frames = loader.scene_frames_dicts[SCENE_TOKEN]
    current_frame = scene_frames[current_idx]
    annotations_dict = current_frame["anns"]
    
    boxes = annotations_dict["gt_boxes"]
    names = annotations_dict["gt_names"]
    instance_tokens = annotations_dict["instance_tokens"]
    
    print(f"\n📦 Annotations: {len(boxes)} objects")
    unique, counts = np.unique(names, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   - {cls}: {cnt}")
    
    # Get camera extrinsics
    R = camera.sensor2lidar_rotation
    t = camera.sensor2lidar_translation
    K = camera.intrinsics
    
    # ========================================================================
    # Test 1: Semantic Segmentation
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Semantic Segmentation from 3D Boxes")
    print("=" * 80)
    
    # Debug first box
    if len(boxes) > 0:
        test_box = boxes[0]
        test_corners = get_box_corners_3d(test_box)
        test_2d, test_depths, test_valid = project_3d_to_2d(test_corners, K, R, t, debug=True)
        print(f"   First box: {test_valid.sum()}/8 corners visible")
    
    semantic_mask, sem_coverage, sem_visible = generate_semantic_mask_from_boxes(
        boxes, names, K, R, t, (IMG_HEIGHT, IMG_WIDTH)
    )
    
    print(f"\n✅ Semantic mask: {semantic_mask.shape}")
    print(f"   - Coverage: {sem_coverage:.1f}%")
    print(f"   - Visible: {sem_visible}/{len(boxes)}")
    
    # ========================================================================
    # Test 2: Depth from LiDAR
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Depth Map from LiDAR")
    print("=" * 80)
    
    depth_map, depth_valid, depth_coverage = generate_depth_from_lidar(
        lidar_pc, K, R, t, (IMG_HEIGHT, IMG_WIDTH)
    )
    
    print(f"\n✅ Depth map: {depth_map.shape}")
    print(f"   - Coverage: {depth_coverage:.1f}%")
    if depth_valid.sum() > 0:
        print(f"   - Range: [{depth_map[depth_valid].min():.1f}, {depth_map[depth_valid].max():.1f}]m")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSemantic:  {sem_coverage:5.1f}% ({sem_visible}/{len(annotations.boxes)} objects)")
    print(f"Depth:     {depth_coverage:5.1f}%")
    
    if sem_coverage > 30 and depth_coverage > 20:
        print("\n🎉 FULL MULTITASK IS FEASIBLE!")
        print("   Proceed with implementation!")
    elif depth_coverage > 20:
        print("\n⚠️  DEPTH-ONLY MULTITASK")
        print("   Semantic coverage too low, but depth works!")
    else:
        print("\n❌ MULTITASK NOT VIABLE")
        print("   Insufficient coverage")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
