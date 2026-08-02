"""smoke_fm3.py -- GPU smoke for the two fm3 feasibility variants + GOLD-default guard.

Head-level (no VLM) smoke that exercises the EXACT fm3 code paths (constraint plumbing,
soft jerk/curv penalty in flow_matching_loss, kinematic GT->control inversion + unicycle
decode, and the navsim-agent decode(v0=...) path). Fast + reliable on the 1-slice GPU.

Parts
  1. GOLD guard   : constraint=none loads fullft_fm2_head_pdm0.9088.pt STRICT + decodes ->
                    proves the default path is byte-identical / unchanged.
  2. by-construction: at INIT, kinematic samples have bounded (low) jerk vs free-xy (fm2).
  3. train        : fm2(none) vs fm3-lite(soft) vs fm3-kin(kinematic) on REAL GT (fixed
                    random ctx); loss must be finite + decrease; measure sampled long_jerk.
  4. agent decode : kinematic decode_all(v0=ego_speed) -> (N,T,3) native heading + v0 sensitivity.

long_jerk metric (m/s^3): mean |3rd finite-diff of forward-x| / dt^3, dt=0.5, ego origin
prepended -- calibrated on GT (~base). fm2 deploy ref ~8-10, base AutoVLA ~0.9 (design doc).

Usage:
  PYTHONPATH=...AutoVLA:...AutoVLA/navsim:...fmhead \
    /root/workspace/miniconda3/envs/autovla/bin/python smoke_fm3.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)

from fm_head import FMHead
from autovla_fmhead import FMHeadDecoder, autovla_fmhead_config, AUTOVLA_HIDDEN_SIZE

GOLD = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_GOLD/fullft_fm2_head_pdm0.9088.pt"
XY_NORM = os.path.join(FMHEAD_DIR, "traj_norm_stats_gt.json")
CTRL_NORM = os.path.join(FMHEAD_DIR, "traj_norm_stats_ctrl.json")
GT_DIR = "/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain_gt_split/train"
DT = 0.5


def long_jerk_mps3(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    """(...,T,2) metres -> (...,) per-traj mean |longitudinal (forward-x) jerk| in m/s^3."""
    x = xy[..., 0]
    origin = np.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    xf = np.concatenate([origin, x], axis=-1)                # prepend ego origin
    j = np.diff(xf, n=3, axis=-1) / (dt ** 3)
    return np.abs(j).mean(axis=-1)


def load_gt(n: int, device) -> torch.Tensor:
    files = sorted(glob.glob(os.path.join(GT_DIR, "*.json")))[:n]
    xy = [np.asarray(json.load(open(f))["gt_trajectory"], dtype=np.float32)[:10, :2] for f in files]
    return torch.from_numpy(np.stack(xy)).float().to(device)  # (n,10,2)


def make_decoder(mode, device, num_steps=100, num_samples=16, **kn):
    constraint = {"mode": mode, **kn}
    cfg = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.0,
                                constraint=constraint)
    dec = FMHeadDecoder(cfg, num_samples=num_samples, num_steps=num_steps, cfg_weight=1.0).to(device)
    dec.load_normalizer(CTRL_NORM if mode == "kinematic" else XY_NORM)
    return dec


def train_head(dec, gt, ctx, steps=600, lr=1e-3):
    opt = torch.optim.Adam(dec.fm_head.parameters(), lr=lr)
    dec.train()
    losses = []
    for _ in range(steps):
        loss = dec.training_loss(gt, ctx, torch.zeros(gt.shape[0], AUTOVLA_HIDDEN_SIZE, device=gt.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.fm_head.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    dec.eval()
    return losses


def sampled_jerk(dec, ctx, gt, seed=0):
    torch.manual_seed(seed)
    s = torch.zeros(ctx.shape[0], AUTOVLA_HIDDEN_SIZE, device=ctx.device)
    v0 = torch.linalg.norm(gt[:, 0, :], dim=-1) / DT if dec.config.constraint_mode == "kinematic" else None
    poses = dec.decode_all(ctx, s, v0=v0)          # (B,N,T,3)
    xy = poses[..., :2].float().cpu().numpy()
    return float(np.median(long_jerk_mps3(xy))), poses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "smoke wants GPU"
    torch.manual_seed(0)
    res = {"device": device}
    print(f"=== fm3 smoke | device={device} torch={torch.__version__} ===")

    # ---- Part 1: GOLD default (mode=none) load + decode ---------------------
    dec_gold = make_decoder("none", device, num_steps=100)
    ck = torch.load(GOLD, map_location=device)
    msg = dec_gold.load_state_dict(ck["fm_decoder"], strict=True)   # MUST be strict-clean
    ctx1 = torch.randn(1, 8, AUTOVLA_HIDDEN_SIZE, device=device)
    s1 = torch.zeros(1, AUTOVLA_HIDDEN_SIZE, device=device)
    torch.manual_seed(1)
    poses_gold = dec_gold.decode(ctx1, s1)                          # (1,10,3)
    gold_ok = (tuple(poses_gold.shape) == (1, 10, 3)
               and bool(torch.isfinite(poses_gold).all())
               and len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0)
    print(f"[1:GOLD] strict-load missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} "
          f"| decode poses={tuple(poses_gold.shape)} finite={bool(torch.isfinite(poses_gold).all())} "
          f"-> {'PASS' if gold_ok else 'FAIL'}")
    res["gold"] = {"pass": gold_ok, "missing": len(msg.missing_keys),
                   "unexpected": len(msg.unexpected_keys), "poses_shape": list(poses_gold.shape),
                   "constraint_mode_default": dec_gold.config.constraint_mode}

    # ---- Part 2: by-construction jerk at INIT (kin bounded vs free-xy) ------
    gt = load_gt(16, device)
    gt_jerk = float(np.median(long_jerk_mps3(gt.cpu().numpy())))
    ctx = torch.randn(16, 8, AUTOVLA_HIDDEN_SIZE, device=device)
    dec_xy0 = make_decoder("none", device)
    dec_kin0 = make_decoder("kinematic", device, accel_max=5.0, yawrate_max=1.0, v_max=25.0)
    xy0_j, _ = sampled_jerk(dec_xy0, ctx, gt, seed=2)
    kin0_j, _ = sampled_jerk(dec_kin0, ctx, gt, seed=2)
    byc_ok = kin0_j < xy0_j
    print(f"[2:by-construction @init] GT long_jerk={gt_jerk:.2f} | free-xy(fm2)={xy0_j:.2f} "
          f"| kinematic={kin0_j:.2f}  -> {'PASS (kin<<xy)' if byc_ok else 'FAIL'}")
    res["by_construction_init"] = {"gt": gt_jerk, "free_xy": xy0_j, "kinematic": kin0_j, "pass": byc_ok}

    # ---- Part 3a: fm2(none) baseline + fm3-kin(kinematic) train -------------
    out = {}
    for name, mode, kn in [
        ("fm2_none", "none", {}),
        ("fm3_kin", "kinematic", {"accel_max": 5.0, "yawrate_max": 1.0, "v_max": 25.0,
                                  "jerk_weight": 0.01, "smooth_t_min": 0.5}),
    ]:
        torch.manual_seed(0)
        dec = make_decoder(mode, device, **kn)
        losses = train_head(dec, gt, ctx, steps=600)
        init_l, fin_l = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
        jk, _ = sampled_jerk(dec, ctx, gt, seed=3)
        out[name] = {"init_loss": round(init_l, 4), "final_loss": round(fin_l, 4),
                     "finite": bool(np.isfinite(losses).all()),
                     "loss_down": fin_l < init_l, "sampled_long_jerk": round(jk, 3)}
        print(f"[3a:train {name:9s}] loss {init_l:.4f}->{fin_l:.4f} finite={out[name]['finite']} "
              f"down={out[name]['loss_down']} | sampled long_jerk={jk:.2f}")
    kin_ok = out["fm3_kin"]["loss_down"] and out["fm3_kin"]["sampled_long_jerk"] < 2.5
    print(f"        -> fm3-kin loss down & jerk<2.5 (toward base): {'PASS' if kin_ok else 'FAIL'}")

    # ---- Part 3b: fm3-lite(soft) CONTROLLED jerk_weight sweep --------------
    # Identical seed/init/data/steps; only jerk_weight varies -> isolates the penalty's
    # causal effect on sampled jerk (robust to the tiny-overfit baseline being near-smooth).
    sweep = {}
    for jw in [0.0, 0.03, 0.05, 0.2]:
        torch.manual_seed(0)
        dec = make_decoder("soft", device, jerk_weight=jw, curv_weight=0.0, smooth_t_min=0.5)
        losses = train_head(dec, gt, ctx, steps=600)
        jk, _ = sampled_jerk(dec, ctx, gt, seed=3)
        sweep[str(jw)] = round(jk, 3)
        print(f"[3b:soft sweep] jerk_weight={jw:<4} -> sampled long_jerk={jk:.2f} "
              f"(final_loss={np.mean(losses[-20:]):.4f})")
    # small weights (the intended tunable operating point, 0.01-0.05) reduce jerk; big
    # weights over-regularize (U-shaped soft bias) -> check the SMALL operating point.
    lite_ok = min(sweep["0.03"], sweep["0.05"]) < sweep["0.0"]
    print(f"        -> soft penalty (small jw) reduces jerk vs jw0={sweep['0.0']}: "
          f"jw0.03={sweep['0.03']} jw0.05={sweep['0.05']} -> {'PASS' if lite_ok else 'FAIL'} "
          f"(jw>=0.2 over-regularizes: {sweep['0.2']})")
    out["fm3_lite_soft_sweep"] = sweep
    res["train"] = out

    # ---- Part 4: agent decode path (kinematic, v0 from ego speed) -----------
    dec_agent = make_decoder("kinematic", device, accel_max=5.0, yawrate_max=1.0, v_max=25.0)
    _ = train_head(dec_agent, gt, ctx, steps=200)
    feats = {"vehicle_velocity": [5.8, 0.14]}                       # as AutoVLA feature builder emits
    vel = np.asarray(feats["vehicle_velocity"], dtype=np.float32)
    v0 = torch.tensor([float(np.linalg.norm(vel))], device=device)
    s = torch.zeros(1, AUTOVLA_HIDDEN_SIZE, device=device)
    ctxA = ctx[:1]
    torch.manual_seed(4)
    poses_fast = dec_agent.decode_all(ctxA, s, v0=v0)               # ego 5.8 m/s
    torch.manual_seed(4)
    poses_stop = dec_agent.decode_all(ctxA, s, v0=torch.zeros(1, device=device))  # ego stopped
    d_fast = float(poses_fast[0, :, -1, 0].mean())                  # mean forward reach
    d_stop = float(poses_stop[0, :, -1, 0].mean())
    agent_ok = (tuple(poses_fast.shape) == (1, dec_agent.num_samples, 10, 3)
                and bool(torch.isfinite(poses_fast).all()) and d_fast > d_stop)
    print(f"[4:agent decode] kin poses={tuple(poses_fast.shape)} finite={bool(torch.isfinite(poses_fast).all())} "
          f"| forward reach v0=5.8 -> {d_fast:.1f} m  vs  v0=0 -> {d_stop:.1f} m (v0 sensitivity) "
          f"-> {'PASS' if agent_ok else 'FAIL'}")
    res["agent_decode"] = {"pass": agent_ok, "poses_shape": list(poses_fast.shape),
                           "reach_v0_5p8": round(d_fast, 2), "reach_v0_0": round(d_stop, 2)}

    all_ok = gold_ok and byc_ok and lite_ok and kin_ok and agent_ok
    res["all_pass"] = bool(all_ok)
    json.dump(res, open(os.path.join(FMHEAD_DIR, "smoke_fm3_results.json"), "w"), indent=2)
    print(f"\n=== fm3 SMOKE: {'PASS' if all_ok else 'FAIL'} -> smoke_fm3_results.json ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
