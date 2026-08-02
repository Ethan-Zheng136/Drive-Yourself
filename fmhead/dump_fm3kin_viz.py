"""dump_fm3kin_viz.py -- dump fm3-kin ep6 (KINEMATIC decode) ego trajectories per token,
in the SAME `{token: [[x_forward, y_left], ...]}` format the 4-panel plotter consumes
(plot_4panel_091_094.py). This is the RIGHT column of the 0.91-vs-fm3-kin comparison.

Why a dedicated dumper (not dump_fmhead_style.py): the pdm-select branch in
dump_fmhead_style.decode_token calls `decode_all(ctx, s, ctx_mask=ctx_mask)` WITHOUT v0.
For the fm3-kin control-space (kinematic-unicycle) head that is a silent correctness bug:
v0=None -> integrate_unicycle starts every scene from a standstill, so the decoded xy is
wrong (and jerk/feasibility no longer reflect the deployed decode). The fm3-kin PDMS run
(0.915) used the AGENT's compute_trajectory, which DOES pass v0 = |ego velocity| from
vehicle_velocity. This dumper replicates that deployed decode exactly (scene-free pdm-select
on the metric-cache token), so the viz shows the SAME trajectories that scored 0.915.

Config (matches run_fm3_pdms.sh fm3-kin block):
  BASE_CKPT   = fullft_fm3_kin_ep6_base.ckpt
  FMHEAD_CKPT = fullft_fm3_kin_ep6_head.pt
  FMHEAD_NORM = traj_norm_stats_ctrl.json           (CONTROL-space z-score)
  CONSTRAINT  = {mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}
  SELECT=pdm (deployed), N=16, STEPS=100, style OFF (s=0), v0 from vehicle_velocity.

Env:
  TOKENS      json list of tokens (default viz_tokens_240.json)   N_TOKENS (default 200)
  OUT_JSON    output dump path (default .../viz_4panel_091_vs_fm3kin_ep6/right/fm3kin_ep6.json)
  FMHEAD_N (16)  FMHEAD_STEPS (100)  METRIC_CACHE (navtest_v1)
Data roots are exported by the launcher (run_dump_fm3kin_viz.sh, from eval_persona.sh).
"""
import os
import json

import numpy as np
import torch
import yaml
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD = "/root/workspace/fmhead"
KIN_DIR = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42"

CONFIG = os.environ.get("CONFIG", os.path.join(REPO, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
BASE_CKPT = os.environ.get("BASE_CKPT", os.path.join(KIN_DIR, "fullft_fm3_kin_ep6_base.ckpt"))
FMHEAD_CKPT = os.environ.get("FMHEAD_CKPT", os.path.join(KIN_DIR, "fullft_fm3_kin_ep6_head.pt"))
FMHEAD_NORM = os.environ.get("FMHEAD_NORM", os.path.join(FMHEAD, "traj_norm_stats_ctrl.json"))
JSON_DIR = os.path.join(REPO, "dataset/nuplan/navtest_nocot")
TOKENS_JSON = os.environ.get("TOKENS", os.path.join(FMHEAD, "viz_tokens_240.json"))
N_TOKENS = int(os.environ.get("N_TOKENS", "200"))
OUT_JSON = os.environ.get(
    "OUT_JSON",
    "/mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_vs_fm3kin_ep6/right/fm3kin_ep6.json")
NSAMP = int(os.environ.get("FMHEAD_N", "16"))
NSTEPS = int(os.environ.get("FMHEAD_STEPS", "100"))
METRIC_CACHE = os.environ.get("METRIC_CACHE", "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
CONSTRAINT = {"mode": "kinematic", "accel_max": 5.0, "yawrate_max": 1.0, "v_max": 25.0, "dt": 0.5}


def jerk_stats(xy, dt=0.5):
    """Longitudinal jerk proxy: 3rd finite difference of the cumulative xy path (m/s^3).
    Returns (max_abs_jerk, rms_jerk) over the trajectory. Used only for a smoke sanity read
    (a kinematically-feasible fm3-kin path should have far lower jerk than 0.91's kinks)."""
    p = np.asarray(xy, dtype=float)
    if p.shape[0] < 4:
        return float("nan"), float("nan")
    vel = np.diff(p, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jrk = np.diff(acc, axis=0) / dt
    mag = np.linalg.norm(jrk, axis=-1)
    return float(mag.max()), float(np.sqrt((mag ** 2).mean()))


def main():
    os.chdir(REPO)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    import sys
    sys.path.insert(0, FMHEAD)
    from fmhead_navsim_agent import FMHeadAutoVLAAgent
    from navsim.common.dataclasses import Trajectory

    all_tokens = json.load(open(TOKENS_JSON))[:N_TOKENS]
    tokens = [t for t in all_tokens if os.path.exists(os.path.join(JSON_DIR, t + ".json"))]
    print(f"[fm3kin-viz] {len(tokens)}/{len(all_tokens)} tokens have navtest_nocot json | "
          f"N={NSAMP} steps={NSTEPS} select=pdm constraint={CONSTRAINT}", flush=True)
    print(f"[fm3kin-viz] base={BASE_CKPT}\n            head={FMHEAD_CKPT}\n            norm={FMHEAD_NORM}", flush=True)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    ts = TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                            interval_length=cfg["model"]["trajectory"]["interval_length"])
    agent = FMHeadAutoVLAAgent(
        trajectory_sampling=ts, checkpoint_path=BASE_CKPT, sensor_data_path=".",
        config_path=CONFIG, lora_conf={"use_lora": False}, device="cuda",
        fmhead_ckpt_path=FMHEAD_CKPT, fmhead_normalizer_path=FMHEAD_NORM,
        fmhead_style_off=True, fmhead_style_alpha=0.0,
        fmhead_num_samples=NSAMP, fmhead_num_steps=NSTEPS, fmhead_cfg_weight=1.0,
        fmhead_select_mode="pdm", fmhead_metric_cache_path=METRIC_CACHE,
        fmhead_use_style_adapter=False, fmhead_constraint=CONSTRAINT)
    agent.initialize()
    agent.autovla.eval()
    assert agent._fm_decoder.config.constraint_mode == "kinematic", "constraint not kinematic!"

    np_poses = ts.num_poses

    def decode_token(tok):
        """Deployed scene-free pdm-select decode WITH v0 (matches compute_trajectory)."""
        ai = json.load(open(os.path.join(JSON_DIR, tok + ".json")))
        features = agent._build_features(ai)
        ctx, ctx_mask, s = agent._extract_ctx_s(features)
        v0 = agent._v0_from_features(features)               # |ego velocity|, kinematic mode
        cands = agent._fm_decoder.decode_all(
            ctx, s, ctx_mask=ctx_mask, v0=v0)[0].float().cpu().numpy()   # (N,T,3)
        scores = []
        for k in range(cands.shape[0]):
            try:
                tr = Trajectory(cands[k][:np_poses], ts)
                scores.append(float(agent._pdm.rl_pdm_score(tr, tok)))
            except Exception as e:  # noqa: BLE001 - a failed candidate must not win
                scores.append(-1.0)
        best = cands[int(np.argmax(scores))][:np_poses, :2]
        return best, float(v0.item()) if v0 is not None else float("nan")

    res, fail = {}, []
    jmax_all, jrms_all, v0_all = [], [], []
    for i, tok in enumerate(tokens):
        try:
            with torch.no_grad():
                xy, v0v = decode_token(tok)
            res[tok] = np.asarray(xy).astype(float).tolist()
            jm, jr = jerk_stats(xy, dt=CONSTRAINT["dt"])
            jmax_all.append(jm); jrms_all.append(jr); v0_all.append(v0v)
        except Exception as e:  # noqa: BLE001 - report, never silently drop
            fail.append(tok)
            if len(fail) <= 5:
                print(f"  fail {tok}: {e!r}"[:200], flush=True)
        if (i + 1) % 20 == 0:
            print(f"[fm3kin-viz] {i+1}/{len(tokens)} decoded ({len(fail)} fail)", flush=True)

    jmax_all = np.array(jmax_all); jrms_all = np.array(jrms_all); v0_all = np.array(v0_all)
    meta = {
        "model": "fm3-kin ep6 (kinematic decode)", "horizon_s": float(ts.time_horizon),
        "n_samples": NSAMP, "steps": NSTEPS, "select": "pdm", "style_off": True,
        "constraint": CONSTRAINT, "n_pts": np_poses, "n_fail": len(fail),
        "base_ckpt": BASE_CKPT, "head_ckpt": FMHEAD_CKPT, "norm": FMHEAD_NORM,
        "frame": "ego BEV, x=forward(m), y=left(m), cumulative positions",
        "jerk_max_mean": float(np.nanmean(jmax_all)) if len(jmax_all) else None,
        "jerk_max_p90": float(np.nanpercentile(jmax_all, 90)) if len(jmax_all) else None,
        "jerk_rms_mean": float(np.nanmean(jrms_all)) if len(jrms_all) else None,
        "v0_mean": float(np.nanmean(v0_all)) if len(v0_all) else None,
    }
    out = {"_meta": meta}
    out.update(res)
    json.dump(out, open(OUT_JSON, "w"))
    print(f"[fm3kin-viz] DONE {len(res)}/{len(tokens)} ok ({len(fail)} fail) -> {OUT_JSON}", flush=True)
    if len(jmax_all):
        print(f"[fm3kin-viz] SANITY jerk(m/s^3): max mean={np.nanmean(jmax_all):.2f} "
              f"p90={np.nanpercentile(jmax_all,90):.2f} max={np.nanmax(jmax_all):.2f} | "
              f"rms mean={np.nanmean(jrms_all):.2f} | v0 mean={np.nanmean(v0_all):.2f} m/s", flush=True)
    if fail:
        print(f"[fm3kin-viz] FAILED tokens ({len(fail)}): {fail[:10]}{' ...' if len(fail)>10 else ''}", flush=True)


if __name__ == "__main__":
    main()
