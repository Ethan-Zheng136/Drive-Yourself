#!/usr/bin/env python
"""build_kth_submission.py -- NEW, additive.

Build a v2 navsim submission.pkl that, for every token, uses the k-th FMHead candidate
trajectory dumped by the agent's `dump_all` mode (candidate_dump_dir/{token}.npy, shape
(N, num_poses, 3)). Scoring each k-th submission with the v2 EPDMS pipeline gives a
per-token score for candidate k; doing this for all k lets us measure the selection
headroom (oracle ceiling) and any candidate's score OFFLINE from ONE inference pass.

Usage:
  python build_kth_submission.py <candidates_dir> <k> <out_submission.pkl>
"""
import sys, glob, os, pickle
import numpy as np

sys.path.insert(0, "/root/workspace/closed_loop/navsim")
from navsim.common.dataclasses import Trajectory  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


def main():
    cand_dir, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    npys = glob.glob(os.path.join(cand_dir, "*.npy"))
    if not npys:
        raise SystemExit(f"ERROR: no *.npy candidate files in {cand_dir}")
    sub, skipped = {}, 0
    for p in npys:
        tok = os.path.basename(p)[:-4]
        arr = np.load(p)  # (N, T, 3)
        if k >= arr.shape[0]:
            skipped += 1
            continue
        poses = np.asarray(arr[k], dtype=np.float32)
        n = poses.shape[0]
        sub[tok] = Trajectory(poses, TrajectorySampling(time_horizon=float(n) * 0.5, interval_length=0.5))
    out_d = {
        "team_name": "youdrive", "authors": "youdrive", "email": "youdrive@local",
        "institution": "youdrive", "country / region": "cn",
        "first_stage_predictions": [sub], "second_stage_predictions": [{}],
    }
    with open(out, "wb") as f:
        pickle.dump(out_d, f)
    print(f"k={k}: built {len(sub)} tokens (skipped {skipped} with <{k+1} candidates) -> {out}", flush=True)


if __name__ == "__main__":
    main()
