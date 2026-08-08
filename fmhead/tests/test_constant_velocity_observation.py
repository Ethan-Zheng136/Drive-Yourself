"""Unit test for models.utils.score.constant_velocity_observation (pdm_pred core logic).

Verifies that the constant-velocity forecast:
  1. keeps the SAME tokens as the source t=0 occupancy frame,
  2. leaves the t=0 frame unchanged (t=0 forecast == observed current frame),
  3. extrapolates each moving agent by exactly velocity * time at every future frame,
  4. keeps static objects / red-light geometries fixed (velocity 0),
  5. does not mutate the source observation.

Runs with the project env, e.g.:
  PYTHONPATH=<repo>/vla_backbone:<repo>/vla_backbone/navsim \
    /root/workspace/miniconda3/envs/autovla/bin/python -m pytest -q \
    fmhead/tests/test_constant_velocity_observation.py
"""
import copy

from shapely.geometry import box

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from models.utils.score import constant_velocity_observation


class _Vel:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Agent:
    def __init__(self, vx, vy):
        self.velocity = _Vel(vx, vy)


class _FakeObs:
    """Minimal stand-in exposing exactly the attributes the helper reads."""

    def __init__(self, occ_maps, unique_objects, sample_interval, sample_res):
        self._occupancy_maps = occ_maps
        self._unique_objects = unique_objects
        self._sample_interval = sample_interval
        self._observation_sample_res = sample_res
        self._red_light_token = "red_light"
        self._collided_track_ids = []


def _centroid(poly):
    c = poly.centroid
    return (round(c.x, 6), round(c.y, 6))


def _build_obs(n_maps=5):
    # t=0 geometries: a moving agent, a static object, a red light
    poly_agent = box(-1, -1, 1, 1)          # centered at (0,0)
    poly_static = box(9, 9, 11, 11)         # centered at (10,10)
    poly_red = box(4, 4, 6, 6)              # centered at (5,5)
    tokens = ["agent", "static", "red_light_42"]
    geoms = [poly_agent, poly_static, poly_red]
    # every stored map holds the same t=0 geometries; helper only reads maps[0]
    occ_maps = [PDMOccupancyMap(list(tokens), list(geoms)) for _ in range(n_maps)]
    unique = {"agent": _Agent(2.0, -1.0), "static": _Agent(0.0, 0.0)}  # red light not in unique
    return _FakeObs(occ_maps, unique, sample_interval=0.1, sample_res=1)


def test_tokens_preserved_and_length():
    obs = _build_obs(n_maps=5)
    cv = constant_velocity_observation(obs)
    assert len(cv._occupancy_maps) == 5
    for m in cv._occupancy_maps:
        assert set(m.tokens) == {"agent", "static", "red_light_42"}


def test_t0_frame_unchanged():
    obs = _build_obs()
    cv = constant_velocity_observation(obs)
    m0 = cv._occupancy_maps[0]
    assert _centroid(m0["agent"]) == (0.0, 0.0)
    assert _centroid(m0["static"]) == (10.0, 10.0)
    assert _centroid(m0["red_light_42"]) == (5.0, 5.0)


def test_constant_velocity_extrapolation():
    obs = _build_obs(n_maps=6)
    cv = constant_velocity_observation(obs)
    dt = 0.1  # sample_interval * sample_res
    for i, m in enumerate(cv._occupancy_maps):
        t = i * dt
        # moving agent: (2, -1) m/s from origin
        assert _centroid(m["agent"]) == (round(2.0 * t, 6), round(-1.0 * t, 6))
        # static + red light never move
        assert _centroid(m["static"]) == (10.0, 10.0)
        assert _centroid(m["red_light_42"]) == (5.0, 5.0)


def test_source_not_mutated():
    obs = _build_obs()
    before = _centroid(obs._occupancy_maps[3]["agent"])
    _ = constant_velocity_observation(obs)
    after = _centroid(obs._occupancy_maps[3]["agent"])
    assert before == after == (0.0, 0.0)


if __name__ == "__main__":
    test_tokens_preserved_and_length()
    test_t0_frame_unchanged()
    test_constant_velocity_extrapolation()
    test_source_not_mutated()
    print("OK: constant_velocity_observation passes all checks")
