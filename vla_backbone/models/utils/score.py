import os
import copy
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
from shapely.affinity import translate as _shapely_translate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def constant_velocity_observation(obs):
    """Rebuild a PDMObservation as a CONSTANT-VELOCITY forecast from the t=0 frame only.

    The GT observation stored in a metric cache bakes in every agent's *actual future*
    position (sampled 0-5s ahead), which is privileged information a deployed planner
    cannot see. This helper replaces that with a deployable forecast: it reads each
    agent's polygon and velocity AT t=0 (the current frame -- what perception would give
    you) and extrapolates its position at every future occupancy step by a constant-
    velocity motion model. No future ground truth is used. Static objects and red-light
    geometries (velocity 0) stay fixed, exactly as they would under a current-frame
    perception + map prior. The HD map / centerline / route / drivable-area of the cache
    are untouched (map-as-prior, available offline to any deployed system).

    :param obs: source PDMObservation (GT-future occupancy) from a metric cache
    :return: a shallow copy of `obs` whose occupancy maps are constant-velocity forecasts
    """
    maps = getattr(obs, "_occupancy_maps", None)
    if not maps:
        return obs

    o0 = maps[0]  # t=0 occupancy: the only frame a deployed planner actually observes
    red = getattr(obs, "_red_light_token", "red_light")
    unique_objects = getattr(obs, "_unique_objects", None) or {}

    # (token, t=0 polygon, vx, vy) for every object present at the current frame
    entries = []
    for tok in list(o0.tokens):
        poly0 = o0[tok]
        if isinstance(tok, str) and tok.startswith(red):
            vx, vy = 0.0, 0.0  # red lights are static map elements, not forecast
        else:
            agent = unique_objects.get(tok)
            vel = getattr(agent, "velocity", None)
            if vel is not None:
                vx, vy = float(vel.x), float(vel.y)
            else:
                vx, vy = 0.0, 0.0  # static object / no velocity -> stays put under CV
        entries.append((tok, poly0, vx, vy))

    # per-map time step: occupancy maps are spaced observation_sample_res * sample_interval
    dt = float(obs._sample_interval) * float(obs._observation_sample_res)

    new_maps = []
    for i in range(len(maps)):
        t = i * dt
        tokens, polygons = [], []
        for tok, poly0, vx, vy in entries:
            tokens.append(tok)
            if vx == 0.0 and vy == 0.0:
                polygons.append(poly0)
            else:
                polygons.append(_shapely_translate(poly0, xoff=vx * t, yoff=vy * t))
        new_maps.append(PDMOccupancyMap(tokens, polygons))

    new_obs = copy.copy(obs)  # share read-only fields (idx map, unique_objects, initialized)
    new_obs._occupancy_maps = new_maps
    new_obs._collided_track_ids = list(getattr(obs, "_collided_track_ids", []))
    return new_obs

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Scene, Trajectory


class PDM_Reward:
    """
    A class that encapsulates the RL PDM reward calculation for gievn token.
    """
    def __init__(self, metric_cache_path, constant_velocity=False):
        """
        Initialize the reward calculator with the given configuration.

        :param metric_cache_path: Path to the metric cache.
        :param constant_velocity: if True, score candidates against a DEPLOYABLE
            constant-velocity forecast of other agents (built from the current frame
            only) instead of the cache's privileged GT-future occupancy. The HD map /
            centerline / route / progress reference are kept (map-as-prior). This backs
            the agent's `mode=pdm_pred` selector -- no future ground truth is consumed.
        """
        # Initialize the necessary components
        self.metric_cache_loader = MetricCacheLoader(metric_cache_path)
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.future_sampling)
        self.scorer= PDMScorer(self.future_sampling)
        self.constant_velocity = constant_velocity

    def rl_pdm_score(self, trajectory, token):
        """
        Compute the rl pdm reward for a given token using the pdm_score metrics, excluding the two_frame_extended_comfort metric.

        :param trajectory: model output.
        :param token: The scene token.
        """
        metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        # pdm_pred: swap the GT-future agent occupancy for a current-frame constant-
        # velocity forecast so selection uses no privileged future information.
        if self.constant_velocity:
            metric_cache.observation = constant_velocity_observation(metric_cache.observation)

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
