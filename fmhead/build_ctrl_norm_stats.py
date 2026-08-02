"""build_ctrl_norm_stats.py -- fit per-(waypoint,dim) stats for the KINEMATIC (variant B)
control space (a_long, yaw_rate).

fm2 flows over normalized xy (traj_norm_stats_gt.json). fm3-kin flows over unicycle
CONTROLS: we invert every GT xy trajectory to (a_long, yaw_rate) with the SAME inversion
the trainer uses (fm_head.FMHead.xy_to_controls, v0 = |wp0|/dt = first_gt_step), then
z-score them per (waypoint, channel). Output format is identical to traj_norm_stats_gt.json
so FMHeadDecoder.load_normalizer consumes it unchanged.

Usage:
  /root/workspace/miniconda3/envs/autovla/bin/python build_ctrl_norm_stats.py \
      [--src DIR] [--out traj_norm_stats_ctrl.json] [--horizon 10] [--dt 0.5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)
from fm_head import FMHead  # noqa: E402

SRC = "/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain12k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(FMHEAD_DIR, "traj_norm_stats_ctrl.json"))
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all files")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.json")))
    assert files, f"no json in {args.src}"
    if args.limit:
        files = files[: args.limit]

    T = args.horizon
    xys = []
    for f in files:
        d = json.load(open(f))
        gt = np.asarray(d["gt_trajectory"], dtype=np.float32)   # (>=T, 3)
        if gt.shape[0] < T:
            continue
        xys.append(gt[:T, :2])
    xy = torch.from_numpy(np.stack(xys, axis=0)).float()        # (N, T, 2)
    v0 = torch.linalg.norm(xy[:, 0, :], dim=-1) / args.dt        # (N,) first_gt_step
    ctrl = FMHead.xy_to_controls(xy, v0, dt=args.dt)            # (N, T, 2) = (a_long, yaw_rate)

    mean = ctrl.mean(dim=0)                                     # (T, 2)
    std = ctrl.std(dim=0).clamp_min(1e-3)                       # (T, 2)
    gmean = ctrl.reshape(-1, 2).mean(dim=0)
    gstd = ctrl.reshape(-1, 2).std(dim=0).clamp_min(1e-3)

    stats = {
        "source": args.src,
        "regime": "human_GT_navtrain_CONTROL_SPACE",
        "space": "unicycle_controls",
        "v0_source": "first_gt_step (|wp0|/dt)",
        "n_samples": int(xy.shape[0]),
        "horizon": T,
        "dt": args.dt,
        "dims": ["a_long_mps2", "yaw_rate_radps"],
        "units": ["m/s^2", "rad/s"],
        "per_waypoint_mean": mean.tolist(),
        "per_waypoint_std": std.tolist(),
        "global_mean": gmean.tolist(),
        "global_std": gstd.tolist(),
    }
    with open(args.out, "w") as fp:
        json.dump(stats, fp, indent=2)

    # quick sanity: round-trip a few samples (invert -> integrate -> compare xy)
    n = min(64, xy.shape[0])
    rec_xy, _ = FMHead.integrate_unicycle(ctrl[:n], v0[:n], dt=args.dt, clamp=False)
    err = torch.linalg.norm(rec_xy - xy[:n], dim=-1).mean().item()
    print(f"[ctrl-norm] n={stats['n_samples']} out={args.out}")
    print(f"[ctrl-norm] a_long  mean/std (global) = {gmean[0]:.4f} / {gstd[0]:.4f} m/s^2")
    print(f"[ctrl-norm] yawrate mean/std (global) = {gmean[1]:.4f} / {gstd[1]:.4f} rad/s")
    print(f"[ctrl-norm] endpoint a_long std = {std[-1,0]:.3f}, yawrate std = {std[-1,1]:.4f}")
    print(f"[ctrl-norm] round-trip xy recon L2 (clamp off) = {err:.2e} m  (should be ~0)")


if __name__ == "__main__":
    main()
