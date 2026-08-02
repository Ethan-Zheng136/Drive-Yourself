from typing import Any, List, Dict, Optional, Union

import torch
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

import tqdm
import pickle
from typing import Dict, Tuple

import torch
import glob
import gzip
import numpy as np
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical

from navsim.agents.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


class VlaAgentFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of AutoVLA Agent."""

    def __init__(self):
        """Initializes the feature builder."""
        pass

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "vla_input_features"

    def compute_features(self, agent_input: AgentInput) -> Dict:
        """Inherited, see superclass."""
        ego_status = agent_input.ego_statuses[-1]
        velocity = ego_status.ego_velocity
        acceleration = ego_status.ego_acceleration

        if np.argmax(ego_status.driving_command) == 0:
            command = 'TURN LEFT'
        elif np.argmax(ego_status.driving_command) == 1:
            command = 'KEEP FORWARD' 
        elif np.argmax(ego_status.driving_command) == 2:
            command = 'TURN RIGHT'
        elif np.argmax(ego_status.driving_command) == 3:
            command = 'UNKNOWN'
        else:
            raise ValueError('Invalid driving command')
        
        history_trajectory_np = np.array([ego_status.ego_pose for ego_status in agent_input.ego_statuses], dtype=np.float32)
        history_trajectory = torch.from_numpy(history_trajectory_np)
        
        front_cam = [agent_input.cameras[i].cam_f0 for i in range(len(agent_input.cameras))]
        front_left_cam = [agent_input.cameras[i].cam_l0 for i in range(len(agent_input.cameras))]
        front_right_cam = [agent_input.cameras[i].cam_r0 for i in range(len(agent_input.cameras))]
        left_camera = [agent_input.cameras[i].cam_l1 for i in range(len(agent_input.cameras))]
        right_camera = [agent_input.cameras[i].cam_r1 for i in range(len(agent_input.cameras))]
        back_camera = [agent_input.cameras[i].cam_b0 for i in range(len(agent_input.cameras))]
        back_left_camera = [agent_input.cameras[i].cam_l2 for i in range(len(agent_input.cameras))]
        back_right_camera = [agent_input.cameras[i].cam_r2 for i in range(len(agent_input.cameras))]

        images = {
            "front_camera": front_cam,
            "front_left_camera": front_left_cam,
            "front_right_camera": front_right_cam,
            "left_camera": left_camera,
            "right_camera": right_camera,
            "back_camera": back_camera,
            "back_left_camera": back_left_camera,
            "back_right_camera": back_right_camera
        }

        features = {
            "vehicle_velocity": velocity,
            "vehicle_acceleration": acceleration,
            "driving_command": command,
            "images": images,
            "history_trajectory": history_trajectory
        }

        return features


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of AutoVLA Agent."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        future_trajectory_tensor = torch.tensor(future_trajectory.poses, dtype=torch.float32)
        return future_trajectory_tensor


class VlaAgent(AbstractAgent):
    """AutoVLA Agent interface."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling
    ):
        """
        Initializes the agent interface for EgoStatusMLP.
        :param trajectory_sampling: trajectory sampling specification.
        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(cam_f0=True, 
                            cam_l0=True, 
                            cam_l1=True, 
                            cam_l2=True, 
                            cam_r0=True, 
                            cam_r1=True, 
                            cam_r2=True,
                            cam_b0=True, 
                            lidar_pc=False)

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [VlaAgentFeatureBuilder()]
