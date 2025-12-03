#!/usr/bin/env python3
"""
Quick test to validate multitask target generation from NAVSIM data.

Tests:
1. Load 3D bounding boxes with semantic classes
2. Project 3D boxes to 2D camera image (semantic + instance)
3. Project LiDAR to depth map
4. Verify coverage and data quality

This validates the approach before full implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pickle
import struct

# ============================================================================
# Configuration
# ============================================================================
SCENE_DIR = Path("sensor_blobs/trainval")  # Base directory, lidar_path includes scene name
METADATA_FILE = Path("navsim_logs/trainval/2021.05.12.19.36.12_veh-35_00005_00204.pkl")
FRAME_IDX = 0  # Test on first frame

# Image dimensions (front camera after cropping/resizing in WoTE)
IMG_HEIGHT = 256
IMG_WIDTH = 1024

# Class mapping (from NAVSIM/nuScenes)
CLASS_MAP = {
    'vehicle': 1,
    'pedestrian': 2,
    'bicycle': 3,
    'traffic_cone': 4,
    'barrier': 5,
    'construction': 6,
    'generic_object': 7,
    # Add more as needed
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_pcd_file(pcd_path):
    """
    Load a PCD (Point Cloud Data) file.
    Handles both ASCII and binary formats.
    
    Returns:
        points: [N, 3] array of (x, y, z) coordinates
    """
    with open(pcd_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line.startswith('DATA'):
                break
        
        # Parse header
        is_ascii = 'ascii' in line.lower()
        
        # Find number of points
        num_points = None
        fields = []
        for line in header_lines:
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('FIELDS'):
                fields = line.split()[1:]
        
        # Read data
        if is_ascii:
            # ASCII format
            data = np.loadtxt(f)
        else:
            # Binary format
            # Assume float32 for x, y, z (typical for PCD)
            data = np.fromfile(f, dtype=np.float32)
            # Reshape based on number of fields
            if len(fields) > 0:
                data = data.reshape(-1, len(fields))
        
        # Extract x, y, z (first 3 columns)
        points = data[:, :3]
        
        return points

def load_metadata(metadata_file):
    """Load scene metadata."""
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    print(f"\n📦 Loaded metadata with {len(metadata)} frames")
    return metadata

def get_box_corners_3d(box):
    """
    Get 8 corners of a 3D bounding box.
    
    Args:
        box: [x, y, z, width, length, height, yaw]
    
    Returns:
        corners: [8, 3] array of corner coordinates
    """
    x, y, z, w, l, h, yaw = box
    
    # Create box in local coordinates (centered at origin)
    # nuScenes/NAVSIM convention: x=forward, y=left, z=up
    corners_local = np.array([
        [-l/2, -w/2, -h/2],  # back-right-bottom
        [-l/2, -w/2,  h/2],  # back-right-top
        [-l/2,  w/2, -h/2],  # back-left-bottom
        [-l/2,  w/2,  h/2],  # back-left-top
        [ l/2, -w/2, -h/2],  # front-right-bottom
        [ l/2, -w/2,  h/2],  # front-right-top
        [ l/2,  w/2, -h/2],  # front-left-bottom
        [ l/2,  w/2,  h/2],  # front-left-top
    ])
    
    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Rotate and translate
    corners_world = corners_local @ R.T + np.array([x, y, z])
    
    return corners_world

def project_3d_to_2d(points_3d, intrinsics, extrinsics):
    """
    Project 3D points in ego/LiDAR frame to 2D camera image.
    
    Args:
        points_3d: [N, 3] points in ego/LiDAR coordinate frame
        intrinsics: [3, 3] camera intrinsic matrix
        extrinsics: dict with 'rotation' [3,3] and 'translation' [3]
                    (sensor2lidar transform)
    
    Returns:
        points_2d: [N, 2] pixel coordinates (u, v)
        depths: [N] depths in camera frame
        valid_mask: [N] boolean mask for valid projections
    """
    # Transform from LiDAR/ego frame to camera frame
    # sensor2lidar gives us: p_lidar = R @ p_sensor + t
    # We want inverse: p_sensor = R^T @ (p_lidar - t)
    R = extrinsics['rotation']
    t = extrinsics['translation']
    
    # Invert transformation
    R_inv = R.T  # Rotation matrix inverse is transpose
    t_inv = -R_inv @ t
    
    # Transform points
    points_cam = (R_inv @ points_3d.T).T + t_inv
    
    # Project to image plane
    # [u, v, d] = K @ [x, y, z]
    points_2d_homogeneous = (intrinsics @ points_cam.T).T  # [N, 3]
    
    # Extract depths (z-coordinate in camera frame)
    depths = points_2d_homogeneous[:, 2]
    
    # Normalize by depth to get pixel coordinates
    valid_depth = depths > 0.1  # Only positive depths in front of camera
    points_2d = np.zeros((len(points_3d), 2))
    points_2d[valid_depth] = points_2d_homogeneous[valid_depth, :2] / depths[valid_depth, None]
    
    # Check if within image bounds
    u, v = points_2d[:, 0], points_2d[:, 1]
    valid_mask = valid_depth & (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
    
    return points_2d.astype(np.int32), depths, valid_mask

def generate_semantic_mask_from_boxes(annotations, intrinsics, extrinsics, img_shape):
    """
    Generate semantic segmentation mask from 3D bounding boxes.
    
    Args:
        annotations: dict with 'gt_boxes' and 'gt_names'
        intrinsics: [3, 3] camera matrix
        extrinsics: dict with rotation and translation
        img_shape: (H, W)
    
    Returns:
        semantic_mask: [H, W] semantic class IDs (0 = background)
        coverage: percentage of pixels labeled
    """
    H, W = img_shape
    semantic_mask = np.zeros((H, W), dtype=np.int32)
    
    boxes = annotations['gt_boxes']
    names = annotations['gt_names']
    
    # Sort by distance (render furthest first for proper occlusion)
    distances = np.linalg.norm(boxes[:, :2], axis=1)
    sorted_indices = np.argsort(-distances)
    
    num_visible = 0
    for idx in sorted_indices:
        box = boxes[idx]
        name = names[idx]
        
        # Get semantic class
        semantic_class = CLASS_MAP.get(name, 0)
        if semantic_class == 0:  # Unknown class
            continue
        
        # Get 8 corners of 3D box
        corners_3d = get_box_corners_3d(box)
        
        # Project to 2D
        corners_2d, depths, valid_mask = project_3d_to_2d(corners_3d, intrinsics, extrinsics)
        
        # Need at least 3 valid corners to draw polygon
        if valid_mask.sum() < 3:
            continue
        
        # Get convex hull of valid projected corners
        valid_corners = corners_2d[valid_mask]
        
        # Rasterize polygon
        cv2.fillPoly(semantic_mask, [valid_corners], int(semantic_class))
        num_visible += 1
    
    # Calculate coverage
    coverage = (semantic_mask > 0).sum() / (H * W) * 100
    
    return semantic_mask, coverage, num_visible

def generate_instance_mask_from_boxes(annotations, intrinsics, extrinsics, img_shape):
    """
    Generate instance segmentation mask from 3D bounding boxes.
    
    Args:
        annotations: dict with 'gt_boxes' and 'instance_tokens'
        intrinsics: [3, 3] camera matrix
        extrinsics: dict with rotation and translation
        img_shape: (H, W)
    
    Returns:
        instance_mask: [H, W] instance IDs (0 = background)
        coverage: percentage of pixels labeled
    """
    H, W = img_shape
    instance_mask = np.zeros((H, W), dtype=np.int32)
    
    boxes = annotations['gt_boxes']
    tokens = annotations['instance_tokens']
    
    # Sort by distance (furthest first)
    distances = np.linalg.norm(boxes[:, :2], axis=1)
    sorted_indices = np.argsort(-distances)
    
    num_visible = 0
    for idx in sorted_indices:
        box = boxes[idx]
        token = tokens[idx]
        
        # Convert token to integer ID (hash and modulo to keep reasonable range)
        instance_id = (hash(token) % 10000) + 1  # +1 to avoid 0 (background)
        
        # Get 8 corners of 3D box
        corners_3d = get_box_corners_3d(box)
        
        # Project to 2D
        corners_2d, depths, valid_mask = project_3d_to_2d(corners_3d, intrinsics, extrinsics)
        
        if valid_mask.sum() < 3:
            continue
        
        valid_corners = corners_2d[valid_mask]
        cv2.fillPoly(instance_mask, [valid_corners], int(instance_id))
        num_visible += 1
    
    coverage = (instance_mask > 0).sum() / (H * W) * 100
    
    return instance_mask, coverage, num_visible

def generate_depth_from_lidar(lidar_pc, intrinsics, extrinsics, img_shape):
    """
    Generate depth map from LiDAR point cloud.
    
    Args:
        lidar_pc: [N, 3] LiDAR points in ego frame
        intrinsics: [3, 3] camera matrix
        extrinsics: dict with rotation and translation
        img_shape: (H, W)
    
    Returns:
        depth_map: [H, W] depth values (0 = no measurement)
        valid_mask: [H, W] boolean mask for valid depths
        coverage: percentage of pixels with depth
    """
    H, W = img_shape
    depth_map = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)
    
    # Project LiDAR points to image
    points_2d, depths, point_valid = project_3d_to_2d(lidar_pc, intrinsics, extrinsics)
    
    # Filter valid points
    valid_points = points_2d[point_valid]
    valid_depths = depths[point_valid]
    
    # Assign depths to pixels (keep closest depth for each pixel)
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
    print("MULTITASK TARGET GENERATION TEST")
    print("=" * 80)
    
    # Load metadata
    if not METADATA_FILE.exists():
        print(f"❌ Metadata file not found: {METADATA_FILE}")
        print("   Please update METADATA_FILE path in the script.")
        return
    
    metadata = load_metadata(METADATA_FILE)
    frame_data = metadata[FRAME_IDX]
    
    # Get camera info (front camera - CAM_F0)
    cam_f0_path = frame_data['cams']['CAM_F0']['data_path']
    cam_intrinsics = np.array(frame_data['cams']['CAM_F0']['cam_intrinsic'])
    cam_extrinsics = {
        'rotation': np.array(frame_data['cams']['CAM_F0']['sensor2lidar_rotation']),
        'translation': np.array(frame_data['cams']['CAM_F0']['sensor2lidar_translation'])
    }
    
    print(f"\n📷 Camera: {cam_f0_path}")
    print(f"   Intrinsics shape: {cam_intrinsics.shape}")
    print(f"   Focal length: {cam_intrinsics[0, 0]:.1f}px")
    
    # Get annotations
    annotations = frame_data['anns']
    num_objects = len(annotations['gt_boxes'])
    print(f"\n📦 Annotations: {num_objects} objects")
    unique_classes, counts = np.unique(annotations['gt_names'], return_counts=True)
    for cls, cnt in zip(unique_classes, counts):
        print(f"   - {cls}: {cnt}")
    
    # Get LiDAR
    lidar_path = frame_data['lidar_path']
    lidar_full_path = SCENE_DIR / lidar_path
    
    # Check if file exists, otherwise use first available
    if not lidar_full_path.exists():
        print(f"⚠️  LiDAR file from metadata not found: {lidar_path}")
        scene_lidar_dir = Path("sensor_blobs/trainval/2021.05.12.19.36.12_veh-35_00005_00204/MergedPointCloud")
        pcd_files = list(scene_lidar_dir.glob("*.pcd"))
        if pcd_files:
            lidar_full_path = pcd_files[0]
            print(f"   Using first available: {lidar_full_path.name}")
        else:
            print(f"❌ No LiDAR files found!")
            return
    
    # Load PCD file
    print(f"\n🔦 Loading LiDAR from: {lidar_full_path.name}")
    lidar_pc = load_pcd_file(lidar_full_path)
    print(f"   Loaded {len(lidar_pc)} points")
    
    # ========================================================================
    # Test 1: Semantic Segmentation from 3D Boxes
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Semantic Segmentation from 3D Boxes")
    print("=" * 80)
    
    semantic_mask, sem_coverage, sem_visible = generate_semantic_mask_from_boxes(
        annotations, cam_intrinsics, cam_extrinsics, (IMG_HEIGHT, IMG_WIDTH)
    )
    
    print(f"✅ Generated semantic mask: {semantic_mask.shape}")
    print(f"   - Coverage: {sem_coverage:.1f}% of pixels")
    print(f"   - Visible objects: {sem_visible}/{num_objects}")
    print(f"   - Unique classes: {np.unique(semantic_mask)}")
    
    # ========================================================================
    # Test 2: Instance Segmentation from 3D Boxes
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Instance Segmentation from 3D Boxes")
    print("=" * 80)
    
    instance_mask, inst_coverage, inst_visible = generate_instance_mask_from_boxes(
        annotations, cam_intrinsics, cam_extrinsics, (IMG_HEIGHT, IMG_WIDTH)
    )
    
    print(f"✅ Generated instance mask: {instance_mask.shape}")
    print(f"   - Coverage: {inst_coverage:.1f}% of pixels")
    print(f"   - Visible objects: {inst_visible}/{num_objects}")
    print(f"   - Unique instances: {len(np.unique(instance_mask)) - 1}")  # -1 for background
    
    # ========================================================================
    # Test 3: Depth from LiDAR
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Depth Map from LiDAR")
    print("=" * 80)
    
    depth_map, depth_valid, depth_coverage = generate_depth_from_lidar(
        lidar_pc, cam_intrinsics, cam_extrinsics, (IMG_HEIGHT, IMG_WIDTH)
    )
    
    print(f"✅ Generated depth map: {depth_map.shape}")
    print(f"   - Coverage: {depth_coverage:.1f}% of pixels")
    print(f"   - Depth range: [{depth_map[depth_valid].min():.1f}, {depth_map[depth_valid].max():.1f}]m")
    print(f"   - Mean depth: {depth_map[depth_valid].mean():.1f}m")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # Semantic
    axes[0, 0].imshow(semantic_mask, cmap='tab10', vmin=0, vmax=10)
    axes[0, 0].set_title(f'Semantic Segmentation\n{sem_coverage:.1f}% coverage, {sem_visible} objects')
    axes[0, 0].axis('off')
    
    # Instance
    axes[0, 1].imshow(instance_mask, cmap='nipy_spectral')
    axes[0, 1].set_title(f'Instance Segmentation\n{inst_coverage:.1f}% coverage, {inst_visible} objects')
    axes[0, 1].axis('off')
    
    # Depth
    depth_vis = depth_map.copy()
    depth_vis[~depth_valid] = np.nan
    im = axes[1, 0].imshow(depth_vis, cmap='turbo', vmin=0, vmax=80)
    axes[1, 0].set_title(f'Depth Map (LiDAR)\n{depth_coverage:.1f}% coverage')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], label='Depth (m)')
    
    # Combined overlay
    # Show semantic with depth overlay
    combined = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    # Semantic as color
    semantic_colored = plt.cm.tab10(semantic_mask / 10.0)[:, :, :3] * 255
    combined = semantic_colored.astype(np.uint8)
    # Depth as transparency
    alpha = (depth_valid * 0.5).astype(np.float32)
    combined = (combined * (1 - alpha[:, :, None])).astype(np.uint8)
    
    axes[1, 1].imshow(combined)
    axes[1, 1].set_title('Combined (Semantic + Depth)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = 'multitask_target_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved visualization: {output_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n📊 Coverage Statistics:")
    print(f"   Semantic:  {sem_coverage:5.1f}% ({sem_visible}/{num_objects} objects visible)")
    print(f"   Instance:  {inst_coverage:5.1f}% ({inst_visible}/{num_objects} objects visible)")
    print(f"   Depth:     {depth_coverage:5.1f}% (LiDAR projection)")
    
    print("\n✅ FEASIBILITY ASSESSMENT:")
    if sem_coverage > 30 and inst_coverage > 30 and depth_coverage > 20:
        print("   🎉 FULL MULTITASK IS FEASIBLE!")
        print("      - All three tasks have sufficient coverage")
        print("      - 3D box projection works for semantic & instance")
        print("      - LiDAR projection works for depth")
        print("\n   📝 Recommendation: Implement full multitask learning!")
    elif depth_coverage > 20:
        print("   ⚠️  DEPTH-ONLY MULTITASK RECOMMENDED")
        print("      - Semantic/instance coverage too low")
        print("      - LiDAR depth is viable")
        print("\n   📝 Recommendation: Implement depth-only multitask")
    else:
        print("   ❌ MULTITASK NOT RECOMMENDED")
        print("      - Insufficient coverage for all tasks")
        print("\n   📝 Recommendation: Focus on ConvGRU only")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
