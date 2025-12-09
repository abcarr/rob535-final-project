#!/usr/bin/env python3
"""
Quick script to list available scene tokens from a specific log.
"""

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig, SceneFilter

# The log file you want to check
LOG_NAME = "2021.05.12.19.36.12_veh-35"

print(f"Loading scenes from log: {LOG_NAME}")
print("=" * 80)

# Create filter for just this log
scene_filter = SceneFilter(
    num_history_frames=4,
    num_future_frames=10,
    has_route=True,
    log_names=[LOG_NAME],
    max_scenes=20,  # First 20 scenes
)

# Load scenes
sensor_config = SensorConfig.build_no_sensors()  # Don't load sensor data
loader = SceneLoader(
    sensor_blobs_path=Path("sensor_blobs/trainval"),
    data_path=Path("navsim_logs/trainval"),
    scene_filter=scene_filter,
    sensor_config=sensor_config,
)

print(f"\nFound {len(loader.tokens)} scenes:")
print("-" * 80)
for i, token in enumerate(loader.tokens):
    print(f"{i+1:2d}. {token}")

print("\n" + "=" * 80)
print(f"Use any of these tokens in your test script!")
print(f"Example: SCENE_TOKEN = \"{loader.tokens[0]}\"")
