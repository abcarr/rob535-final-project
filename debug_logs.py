#!/usr/bin/env python3
"""
Debug script to understand log structure and scene generation.
"""

import pickle
from pathlib import Path

# Check what log files exist
data_path = Path("navsim_logs/trainval")
log_files = sorted(data_path.glob("*.pkl"))

print("=" * 80)
print(f"Found {len(log_files)} log files in {data_path}")
print("=" * 80)

# Show first few log files
print("\nFirst 10 log files:")
for i, log_file in enumerate(log_files[:10]):
    print(f"{i+1:2d}. {log_file.name}")

# Pick the first log and examine its structure
if log_files:
    print("\n" + "=" * 80)
    print(f"Examining: {log_files[0].name}")
    print("=" * 80)
    
    scene_dict_list = pickle.load(open(log_files[0], "rb"))
    print(f"Number of frames in log: {len(scene_dict_list)}")
    
    # Show first frame structure
    if scene_dict_list:
        print(f"\nFirst frame keys: {list(scene_dict_list[0].keys())}")
        print(f"\nFirst frame token: {scene_dict_list[0].get('token', 'NO TOKEN FIELD!')}")
        print(f"Has route? roadblock_ids length: {len(scene_dict_list[0].get('roadblock_ids', []))}")
        
        # Check a few frames
        print(f"\nSample of frame tokens (first 10):")
        for i in range(min(10, len(scene_dict_list))):
            token = scene_dict_list[i].get('token', 'N/A')
            route_len = len(scene_dict_list[i].get('roadblock_ids', []))
            print(f"  Frame {i}: {token} (route_len={route_len})")
        
        # Test scene generation logic
        print("\n" + "=" * 80)
        print("Testing scene generation (num_history=4, num_future=10, total=14 frames)")
        print("=" * 80)
        
        if len(scene_dict_list) >= 14:
            # Check if first 14 frames would make a valid scene
            test_frames = scene_dict_list[:14]
            middle_frame = test_frames[3]  # 4th frame (num_history_frames - 1 = 3)
            scene_token = middle_frame.get('token')
            route_len = len(middle_frame.get('roadblock_ids', []))
            
            print(f"First scene would be:")
            print(f"  Frames: 0-13")
            print(f"  Scene token (from frame 3): {scene_token}")
            print(f"  Has route? {route_len > 0} (route_len={route_len})")
            
            if route_len == 0:
                print("\n⚠️  ISSUE: This scene has no route (has_route filter would exclude it)")
                print("    Try setting has_route=False in SceneFilter")
