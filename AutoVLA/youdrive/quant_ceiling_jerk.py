"""Codebook round-trip JERK ceiling test.

Snaps each teacher's continuous gt_trajectory to the nearest codebook tokens
(deterministic, num_k=1) and DECODES back, then measures how much aggressive
KINEMATIC STYLE (jerk/accel) survives the discrete representation.

If teacher long_jerk (~4.16 for DDv2) collapses to ~the persona/GRPO ceiling
(~2.9) after round-trip  => the fixed action codebook CAPS style; LoRA/RL/
capacity cannot cross it; the fix is a finer codebook or a continuous head.
If round-trip keeps ~4.0  => codebook is NOT the bottleneck.

Usage: python youdrive/quant_ceiling_jerk.py [N_SAMPLE]
"""
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from omegaconf import DictConfig

from navsim.agents.autovla_agent import TokenProcessor

CODEBOOK = "codebook_cache/agent_vocab.pkl"
TEACHERS = {
    "DDv2":       "/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k",
    "GoalFlow":   "/mnt/pfs/zhengguantian/autovla/persona/nocot_goalflow_teacher12k_fixed",
    "TransFuser": "/mnt/pfs/zhengguantian/autovla/persona/nocot_transfuser_teacher12k",
}
OUT = "/mnt/pfs/zhengguantian/autovla/persona/codebook_roundtrip.txt"
DT = 0.5
KEYS = ["long_jerk", "lat_jerk", "peak_acc", "peak_dec", "lat_amax"]


def kin(vx, vy):
    vx = np.asarray(vx, float); vy = np.asarray(vy, float); sp = np.hypot(vx, vy)
    ax = np.diff(sp) / DT if len(sp) > 1 else np.array([0.])
    lj = np.diff(ax) / DT if len(ax) > 1 else np.array([0.])
    ay = np.diff(vy) / DT if len(vy) > 1 else np.array([0.])
    aj = np.diff(ay) / DT if len(ay) > 1 else np.array([0.])
    return {
        "peak_acc": float(ax.max()) if ax.size else 0.0,
        "peak_dec": float(-ax.min()) if ax.size else 0.0,
        "long_jerk": float(np.sqrt((lj ** 2).mean())) if lj.size else 0.0,
        "lat_amax": float(np.abs(ay).max()) if ay.size else 0.0,
        "lat_jerk": float(np.sqrt((aj ** 2).mean())) if aj.size else 0.0,
    }


def mvxvy(xy):
    p = np.array([[0., 0.]] + list(xy), float); d = np.diff(p, axis=0) / DT
    return d[:, 0], d[:, 1]


def med(rows, k):
    return float(np.median([r[k] for r in rows])) if rows else float("nan")


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    tp = TokenProcessor(
        agent_token_file=CODEBOOK,
        agent_token_sampling=DictConfig({"num_k": 1, "temp": 1.0}),
    )
    tp.eval()  # deterministic nearest-token snap = the codebook's representational ceiling

    lines = []
    def emit(s):
        print(s, flush=True); lines.append(s)

    for name, tdir in TEACHERS.items():
        files = sorted(glob.glob(f"{tdir}/*.json"))
        if not files:
            emit(f"[{name}] no teacher json under {tdir} -- SKIP"); continue
        random.seed(0)
        if n_sample and n_sample < len(files):
            files = random.sample(files, n_sample)
        orig_rows, rt_rows = [], []
        n_ok = 0
        for fp in files:
            try:
                traj = json.load(open(fp)).get("gt_trajectory")
                if not traj:
                    continue
                arr = np.asarray(traj, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
                    continue
                with torch.no_grad():
                    out = tp(torch.from_numpy(arr))
                q = out["gt_pos"][0].cpu().numpy()       # [T,2] quantized (round-trip) positions
                o = arr[:q.shape[0], :2]                 # [T,2] original positions
                orig_rows.append(kin(*mvxvy(o)))
                rt_rows.append(kin(*mvxvy(q)))
                n_ok += 1
            except Exception as e:
                emit(f"  [{name}] skip {os.path.basename(fp)}: {e!r}")
                continue
        emit("")
        emit(f"===== {name} teacher  codebook round-trip JERK  (n={n_ok}) =====")
        emit(f"  {'metric':10s} {'original':>10s} {'round-trip':>12s} {'retained%':>10s}")
        for k in KEYS:
            o = med(orig_rows, k); r = med(rt_rows, k)
            ret = (r / o * 100) if o > 1e-9 else float("nan")
            emit(f"  {k:10s} {o:10.3f} {r:12.3f} {ret:9.0f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    emit(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
