import os
import torch
import yaml
import lzma
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Scene, Trajectory


class PDM_Reward:
    """
    A class that encapsulates the RL PDM reward calculation for gievn token.
    """
    def __init__(self, metric_cache_path):
        """
        Initialize the reward calculator with the given configuration.

        :param metric_cache_path: Path to the metric cache.
        """
        # Initialize the necessary components
        self.metric_cache_loader = MetricCacheLoader(metric_cache_path)
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.future_sampling)
        self.scorer= PDMScorer(self.future_sampling)

    def rl_pdm_score(self, trajectory, token):
        """
        Compute the rl pdm reward for a given token using the pdm_score metrics, excluding the two_frame_extended_comfort metric.

        :param trajectory: model output.
        :param token: The scene token.
        """
        metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        try:
            # Compute the pdm score
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.future_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            )

            final_reward = result.score

            return final_reward

        except Exception as e:
            # Surface the token + error: a silent 0.0 here is indistinguishable from a
            # genuinely unsafe rollout and would corrupt the PDMS safety reward signal.
            print(f"[PDM_Reward] rl_pdm_score failed for token={token}: {repr(e)}", flush=True)

            return 0.0
