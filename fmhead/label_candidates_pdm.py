"""label_candidates_pdm.py -- offline PDM/EPDMS labels for FMHead candidates (scorer KD).

For every token dumped by the agent's `dump_scorer` mode ({token}.npz with `cands`
(N, T, 3)), run the REAL NAVSIM pdm_score on each of the N candidates and record its
per-metric sub-scores. These are the knowledge-distillation TARGETS for FMHeadScorer:
the scorer learns to predict this privileged simulator output from inference-available
features, then replaces the simulator at inference.

This is the SAME scoring used by mode=pdm (models.utils.score.PDM_Reward -> pdm_score),
but we read ALL PDMResults fields (not just the scalar .score). No AutoVLA-repo edits.

Output: {out_dir}/{token}.npz with, for each metric key, a (N,) float array in [0,1],
plus `score` (N,) = the final PDMS. v1 metric cache gives NC/DAC/EP/TTC/comfort/DDC; a v2
EPDMS cache additionally enables lane_keeping/traffic_light_compliance (handled by the v2
scorer variant -- not this v1 script).

Run (CPU; PYTHONPATH must include AutoVLA + its navsim + fmhead). Shardable across many
CPU jobs via --num_shards/--shard_index.
"""

from __future__ import annotations

import argparse
import glob
import lzma
import os
import pickle
import sys
import time

import numpy as np

AUTOVLA_ROOT = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
for p in (AUTOVLA_ROOT, os.path.join(AUTOVLA_ROOT, "navsim"),
          os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

# v1 PDMResults sub-metrics (exact field names, see navsim .../dataclasses.py::PDMResults).
METRIC_KEYS = [
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance",
]


def build_scorer():
    from navsim.common.dataloader import MetricCacheLoader  # noqa: E402
    from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer  # noqa: E402
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
    from navsim.common.dataclasses import Trajectory  # noqa: E402
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402
    return dict(pdm_score=pdm_score, PDMScorer=PDMScorer, PDMSimulator=PDMSimulator,
                Trajectory=Trajectory, TrajectorySampling=TrajectorySampling,
                MetricCacheLoader=MetricCacheLoader)


def build_pdm_tools(metric_cache_dir):
    """Build the (reusable) NAVSIM PDM scoring stack ONCE. Shared by the offline labeler
    (main) AND the JOINT trainer (fmhead_sft_joint.py) so both score candidates identically."""
    from pathlib import Path
    M = build_scorer()
    fut = M["TrajectorySampling"](num_poses=40, interval_length=0.1)  # internal PDM sim sampling
    return {
        "M": M,
        "fut": fut,
        "simulator": M["PDMSimulator"](fut),
        "scorer": M["PDMScorer"](fut),
        "loader": M["MetricCacheLoader"](Path(metric_cache_dir)),
    }


def score_candidates(cands, token, tools, interval=0.5):
    """Score N candidate trajectories with the REAL NAVSIM pdm_score.

    cands: (N, T, 3) np.float32 in navsim ego metres. Returns {metric: (N,) float in [0,1]}
    plus 'score' (N,), or None if the token has no metric cache. A per-candidate sim failure
    is LOGGED and that candidate's row is left 0 (never silently swallowed for all)."""
    M, fut = tools["M"], tools["fut"]
    cache_paths = tools["loader"].metric_cache_paths
    if token not in cache_paths:
        return None
    cands = np.asarray(cands, dtype=np.float32)
    n_poses = cands.shape[1]
    samp = M["TrajectorySampling"](time_horizon=float(n_poses) * interval, interval_length=interval)
    with lzma.open(cache_paths[token], "rb") as f:
        metric_cache = pickle.load(f)
    labels = {k: np.zeros(cands.shape[0], dtype=np.float32) for k in METRIC_KEYS}
    labels["score"] = np.zeros(cands.shape[0], dtype=np.float32)
    for k in range(cands.shape[0]):
        try:
            tr = M["Trajectory"](cands[k], samp)
            res = M["pdm_score"](metric_cache=metric_cache, model_trajectory=tr,
                                 future_sampling=fut, simulator=tools["simulator"],
                                 scorer=tools["scorer"])
            for mk in METRIC_KEYS:
                labels[mk][k] = float(getattr(res, mk))
            labels["score"][k] = float(res.score)
        except Exception as e:  # noqa: BLE001 -- log, never silently zero a bad candidate
            print(f"[label] pdm_score failed token={token} cand={k}: {e!r}", flush=True)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="dir of {token}.npz from dump_scorer")
    ap.add_argument("--metric_cache", required=True, help="PDM metric cache dir (e.g. metric_cache_navtrain12k)")
    ap.add_argument("--out_dir", required=True, help="dir to write {token}.npz per-candidate labels")
    ap.add_argument("--interval", type=float, default=0.5, help="candidate pose interval (s)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only first N tokens")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tools = build_pdm_tools(args.metric_cache)
    cache_paths = tools["loader"].metric_cache_paths

    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    files = files[args.shard_index::args.num_shards]
    if args.limit:
        files = files[: args.limit]
    print(f"[label] {len(files)} tokens (shard {args.shard_index}/{args.num_shards}) "
          f"-> {args.out_dir}", flush=True)

    n_ok = n_skip = n_missing = 0
    t0 = time.time()
    for i, fp in enumerate(files):
        token = os.path.splitext(os.path.basename(fp))[0]
        out_fp = os.path.join(args.out_dir, f"{token}.npz")
        if os.path.exists(out_fp) and not args.overwrite:
            n_skip += 1
            continue
        if token not in cache_paths:
            print(f"[label] MISSING metric cache for token={token}", flush=True)
            n_missing += 1
            continue
        cands = np.load(fp)["cands"].astype(np.float32)          # (N, T, 3)
        labels = score_candidates(cands, token, tools, interval=args.interval)
        np.savez(out_fp, **labels)
        n_ok += 1
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[label] {i+1}/{len(files)} ok={n_ok} skip={n_skip} miss={n_missing} "
                  f"({rate:.2f} tok/s)", flush=True)

    print(f"[label] DONE ok={n_ok} skip={n_skip} missing={n_missing} in {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
