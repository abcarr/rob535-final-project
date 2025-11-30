from typing import Any, List, Dict, Union
import torch
import numpy as np
from torchvision import transforms

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import BoundingBoxIndex, LidarIndex

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import timm, cv2

class WoTEFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
            self,
            slice_indices=[3],
            config=None,
        ):
        self.slice_indices = slice_indices
        self._config = config

    def get_unique_name(self) -> str:
        return "wote_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        # Check if multi-frame mode (temporal fusion)
        is_temporal = len(self.slice_indices) > 1

        features["camera_feature"] = self._get_camera_feature(agent_input, is_temporal)
        features["lidar_feature"] = self._get_lidar_feature(agent_input, is_temporal)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        # Add ego-motion transforms for temporal fusion
        if is_temporal:
            features["ego_motion"] = self._compute_ego_motion_transforms(agent_input)

        return features
    
    def _get_camera_feature(self, agent_input: AgentInput, is_temporal: bool = False) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :param is_temporal: whether to extract multi-frame sequence
        :return: stitched front view image as torch tensor
        """
        if is_temporal:
            # Multi-frame extraction for temporal fusion
            camera_frames = []
            for idx in self.slice_indices:
                cameras = agent_input.cameras[idx]
                
                # Crop to ensure 4:1 aspect ratio
                l0 = cameras.cam_l0.image[28:-28, 416:-416]
                f0 = cameras.cam_f0.image[28:-28]
                r0 = cameras.cam_r0.image[28:-28, 416:-416]
                
                # stitch l0, f0, r0 images
                stitched_image = np.concatenate([l0, f0, r0], axis=1)
                resized_image = cv2.resize(stitched_image, (1024, 256))
                tensor_image = transforms.ToTensor()(resized_image)
                camera_frames.append(tensor_image)
            
            # Stack temporal frames: [T, C, H, W]
            return torch.stack(camera_frames, dim=0)
        else:
            # Single-frame extraction (original behavior)
            cameras = agent_input.cameras[-1]

            # Crop to ensure 4:1 aspect ratio
            l0 = cameras.cam_l0.image[28:-28, 416:-416]
            f0 = cameras.cam_f0.image[28:-28]
            r0 = cameras.cam_r0.image[28:-28, 416:-416]

            # stitch l0, f0, r0 images
            stitched_image = np.concatenate([l0, f0, r0], axis=1)
            resized_image = cv2.resize(stitched_image, (1024, 256))
            tensor_image = transforms.ToTensor()(resized_image)

            return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput, is_temporal: bool = False) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :param is_temporal: whether to extract multi-frame sequence
        :return: LiDAR histogram as torch tensors
        """
        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                int((self._config.lidar_max_x - self._config.lidar_min_x)
                * self._config.pixels_per_meter)
                + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                int((self._config.lidar_max_y - self._config.lidar_min_y)
                * self._config.pixels_per_meter)
                + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        def process_single_lidar(lidar_pc):
            """Helper to process a single LiDAR frame."""
            # Remove points above the vehicle
            lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
            below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
            above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
            above_features = splat_points(above)
            if self._config.use_ground_plane:
                below_features = splat_points(below)
                features = np.stack([below_features, above_features], axis=-1)
            else:
                features = np.stack([above_features], axis=-1)
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)
            return torch.tensor(features)

        if is_temporal:
            # Multi-frame extraction for temporal fusion
            lidar_frames = []
            for idx in self.slice_indices:
                # only consider (x,y,z) & swap axes for (N,3) numpy array
                lidar_pc = agent_input.lidars[idx].lidar_pc[LidarIndex.POSITION].T
                lidar_feature = process_single_lidar(lidar_pc)
                lidar_frames.append(lidar_feature)
            
            # Stack temporal frames: [T, C, H, W]
            return torch.stack(lidar_frames, dim=0)
        else:
            # Single-frame extraction (original behavior)
            # only consider (x,y,z) & swap axes for (N,3) numpy array
            lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T
            return process_single_lidar(lidar_pc)
    
    def _compute_ego_motion_transforms(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute 2D affine transformation matrices for BEV warping from historical frames to current frame.
        
        IMPORTANT: In NAVSIM, ego_pose is already relative to the most recent frame (current frame).
        The ego poses are computed via convert_absolute_to_relative_se2_array() in AgentInput.from_scene_dict_list(),
        which means all historical poses are already in the current frame's coordinate system.
        
        This means:
        - ego_pose[0] = delta_x (relative to current frame)
        - ego_pose[1] = delta_y (relative to current frame)  
        - ego_pose[2] = delta_heading (relative to current frame)
        
        Args:
            agent_input: Input dataclass containing ego states at each timestep
        
        Returns:
            Affine matrices [T-1, 2, 3] to warp frames [0..T-2] → frame [T-1]
            Each matrix transforms points from historical frame coordinates to current frame coordinates
        """
        # Get ego states for all temporal frames
        ego_states = [agent_input.ego_statuses[idx] for idx in self.slice_indices]
        
        transforms = []
        
        # Compute transformation for each historical frame to align with current frame
        # Since ego_pose is already relative to the current frame, we can use it directly
        for prev_state in ego_states[:-1]:
            # ego_pose is [x, y, heading] relative to current frame
            rel_x = prev_state.ego_pose[0]
            rel_y = prev_state.ego_pose[1]
            rel_heading = prev_state.ego_pose[2]
            
            # Build 2D affine matrix for BEV warping
            # This transforms points from prev frame to current frame
            cos_h = np.cos(rel_heading)
            sin_h = np.sin(rel_heading)
            
            # Affine transformation matrix [2, 3]: [R | t]
            # where R is rotation and t is translation
            # 
            # VERIFIED from WoTE config:
            # - BEV spatial size: 8×8 (backbone output size)
            # - LiDAR range: [-32, 32] in both x and y (64m total)
            # - pixels_per_meter: 4.0
            # Therefore: BEV covers 64m with 8 pixels → 8/64 = 0.125 pixels per meter
            scale_factor = 8.0 / 64.0  # BEV spatial size (8×8) / physical range (64m)
            
            affine_matrix = np.array([
                [cos_h, -sin_h, rel_x * scale_factor],
                [sin_h,  cos_h, rel_y * scale_factor]
            ], dtype=np.float32)
            
            transforms.append(torch.tensor(affine_matrix))
        
        # Stack into [T-1, 2, 3] tensor
        return torch.stack(transforms, dim=0)
    