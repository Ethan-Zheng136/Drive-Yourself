"""eval_jerk_feasibility.py -- decode-trajectory jerk/feasibility audit for the fm3 heads.

For >=N navtest scenes, decode the FMHead trajectory EXACTLY as deployed (medoid select,
style OFF, s=0; kinematic variants integrate (a_long,yaw_rate)->xy with v0 from ego speed),
then measure per-scene feasibility metrics with the SAME finite-difference convention as
FMHead._smoothness_penalty (xy space, origin (0,0) prepended, dt=0.5):

  long_jerk = mean_t |d^3 x_forward / dt^3|      (m/s^3)
  lat_jerk  = mean_t |d^3 y_left    / dt^3|      (m/s^3)
  peak_acc  = max_t ( d^2 x_forward / dt^2 )     (m/s^2, hardest forward accel)
  peak_dec  = max_t (-d^2 x_forward / dt^2 )     (m/s^2, hardest braking magnitude)
  kink_max  = max_t |wrap(dpsi)| of segment heading (rad/step); acute-kink flag if > 1.0

Reports per-scene values aggregated by MEDIAN (robust) and MEAN across scenes. The human-GT
row (computed from each scene's gt_trajectory) is an anchor: the method is calibrated if GT
long_jerk ~ 0.29 (README/fm3_design anchors: fm2 deploy 8-10, base AutoVLA ~0.9, GT ~0.29).

Runs ONE variant per process (build cost dominated by the VLM). No metric cache / SceneLoader
needed: medoid decode uses only the per-token prompt json (+ its camera symlinks). GPU node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

FM = os.path.dirname(os.path.abspath(__file__))
AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
for p in (FM, AV, AV + "/navsim"):
    if p not in sys.path:
        sys.path.insert(0, p)

DT = 0.5


def feasibility_metrics(poses_xy: np.ndarray):
    """poses_xy: (T,2) metres ego-frame (x_forward, y_left). Returns dict of scalars.

    Matches FMHead._smoothness_penalty (xy): prepend origin, 3x finite diff / dt."""
    P = np.concatenate([np.zeros((1, 2), dtype=np.float64), poses_xy.astype(np.float64)], axis=0)
    vel = np.diff(P, axis=0) / DT           # (T,2)   m/s
    acc = np.diff(vel, axis=0) / DT         # (T-1,2) m/s^2
    jerk = np.diff(acc, axis=0) / DT        # (T-2,2) m/s^3
    long_jerk = float(np.abs(jerk[:, 0]).mean()) if jerk.shape[0] else 0.0
    lat_jerk = float(np.abs(jerk[:, 1]).mean()) if jerk.shape[0] else 0.0
    a_long = acc[:, 0]
    peak_acc = float(a_long.max()) if a_long.size else 0.0
    peak_dec = float((-a_long).max()) if a_long.size else 0.0
    # heading kinks from segment heading (atan2 of velocity), wrapped diff
    psi = np.arctan2(vel[:, 1], vel[:, 0])
    dpsi = np.diff(psi)
    dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi,pi]
    kink_max = float(np.abs(dpsi).max()) if dpsi.size else 0.0
    return dict(long_jerk=long_jerk, lat_jerk=lat_jerk, peak_acc=peak_acc,
                peak_dec=peak_dec, kink_max=kink_max, end_x=float(poses_xy[-1, 0]))


def agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                p90=float(np.percentile(v, 90)), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fm3_lite", "fm3_kin"])
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--fmhead_ckpt", required=True)
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--config_path",
                    default=os.path.join(AV, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--json_data", default=os.path.join(AV, "dataset/nuplan/navtest_nocot"))
    ap.add_argument("--out", default="/mnt/pfs/zhengguantian/autovla/persona/fm3_eval")
    args = ap.parse_args()

    os.chdir(AV)  # so relative camera symlinks (dataset/nuplan/sensor_blobs/...) resolve
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "jerk audit needs a real GPU (vision forward)"
    os.makedirs(args.out, exist_ok=True)

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from fmhead_navsim_agent import FMHeadAutoVLAAgent

    constraint = None
    if args.variant == "fm3_kin":
        constraint = dict(mode="kinematic", accel_max=5.0, yawrate_max=1.0,
                          v_max=25.0, dt=0.5)

    ts = TrajectorySampling(num_poses=10, interval_length=0.5, time_horizon=5.0)
    agent = FMHeadAutoVLAAgent(
        trajectory_sampling=ts, checkpoint_path=args.base_ckpt, sensor_data_path=".",
        config_path=args.config_path, lora_conf={"use_lora": False}, device=device,
        fmhead_ckpt_path=args.fmhead_ckpt, fmhead_normalizer_path=args.normalizer,
        fmhead_style_off=True, fmhead_select_mode="medoid",
        fmhead_num_samples=args.num_samples, fmhead_num_steps=args.num_steps,
        fmhead_constraint=constraint,
    )
    agent.initialize()
    print(f"[jerk] variant={args.variant} constraint={constraint} "
          f"N={args.num_samples} steps={args.num_steps}", flush=True)

    tokens = sorted(f[:-5] for f in os.listdir(args.json_data) if f.endswith(".json"))[:args.n]
    print(f"[jerk] {len(tokens)} tokens from {args.json_data}", flush=True)

    fm_rows, gt_rows = [], []
    t0 = time.time()
    for k, token in enumerate(tokens):
        try:
            agent_input = json.load(open(os.path.join(args.json_data, f"{token}.json")))
            fm_traj, _ = agent.compute_trajectory(agent_input, None)   # medoid (deploy) decode
            fm = np.asarray(fm_traj.poses)[:, :2]                       # (10,2) metres
            fm_rows.append(dict(token=token, **feasibility_metrics(fm)))
            gt = np.asarray(agent_input["gt_trajectory"], dtype=float)[:ts.num_poses, :2]
            gt_rows.append(dict(token=token, **feasibility_metrics(gt)))
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  [{k+1:4d}/{len(tokens)}] {el:6.1f}s "
                      f"({el/(k+1):.2f}s/scene) last long_jerk fm={fm_rows[-1]['long_jerk']:.2f} "
                      f"gt={gt_rows[-1]['long_jerk']:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad scene must not kill the audit
            print(f"  [{k:4d}] {token[:8]} FAILED: {type(e).__name__}: {e}", flush=True)

    if not fm_rows:
        print("[jerk] no scenes succeeded"); return 1

    keys = ["long_jerk", "lat_jerk", "peak_acc", "peak_dec", "kink_max", "end_x"]
    fm_agg = {k: agg(fm_rows, k) for k in keys}
    gt_agg = {k: agg(gt_rows, k) for k in keys}
    acute = float(np.mean([r["kink_max"] > 1.0 for r in fm_rows]))

    print(f"\n===== JERK / FEASIBILITY  variant={args.variant}  n={len(fm_rows)} =====")
    print(f"{'metric':10s} {'FM_median':>10s} {'FM_mean':>9s} {'FM_p90':>8s} "
          f"{'GT_median':>10s} {'GT_mean':>9s}")
    for k in keys:
        print(f"{k:10s} {fm_agg[k]['median']:10.3f} {fm_agg[k]['mean']:9.3f} "
              f"{fm_agg[k]['p90']:8.3f} {gt_agg[k]['median']:10.3f} {gt_agg[k]['mean']:9.3f}")
    print(f"acute-kink fraction (|dpsi|>1.0 rad/step): {acute:.3f}")

    res = dict(variant=args.variant, n=len(fm_rows), base_ckpt=args.base_ckpt,
               fmhead_ckpt=args.fmhead_ckpt, normalizer=args.normalizer,
               num_samples=args.num_samples, num_steps=args.num_steps,
               constraint=constraint, acute_kink_fraction=acute,
               fm=fm_agg, gt=gt_agg)
    outp = os.path.join(args.out, f"jerk_{args.variant}.json")
    json.dump(res, open(outp, "w"), indent=2)
    print(f"[jerk] -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
