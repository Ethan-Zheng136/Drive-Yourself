#!/usr/bin/env bash
# calib_ref_navtest_epdms.sh -- NEW, additive (modifies NO existing file).
#
# Calibration of the navtest ONE-STAGE v2 EPDMS scale, CPU-only (no GPU, no VLM).
# Builds a HUMAN ground-truth submission straight from the navtest v2 metric cache
# (each metric_cache.pkl stores .human_trajectory, 8x3), scores it through the SAME
# score_existing_submission.sh pipeline our model runs use, and prints BOTH:
#   * raw reported EPDMS (buggy: includes undefined two_frame_extended_comfort=0)
#   * CORRECT single-stage EPDMS (excludes two_frame, as _aggregate_pdm_scores does)
#
# Purpose: is our model's 0.83 low (=> headroom / partial bug) or near the metric
# ceiling? Compare against the human-GT upper bound this produces.
#
# Usage (submit on your compute box; ~10-20 min CPU, 0 GPU):
#   bash /root/workspace/fmhead/calib_ref_navtest_epdms.sh
set -uo pipefail

NAVSIM_V2=/root/workspace/closed_loop/navsim
NAVSIM_PY=/root/workspace/miniconda3/envs/navsim/bin/python
SCORE_SH=/root/workspace/closed_loop/navsim_integration/score_existing_submission.sh
V2_CACHE=/root/workspace/closed_loop/data/navsim/exp/metric_cache_navtest
REF=/mnt/pfs/zhengguantian/autovla/persona/v2epdms/ref_human_navtest
mkdir -p "$REF"

echo "[calib] building HUMAN-GT submission from metric cache (CPU)..."
"$NAVSIM_PY" - "$REF" "$V2_CACHE" <<'PY'
import sys, glob, os, pickle, lzma, numpy as np
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
ref, cache = sys.argv[1], sys.argv[2]
pkls = glob.glob(f"{cache}/*/*/*/metric_cache.pkl")
print("num metric caches:", len(pkls), flush=True)
sub, bad = {}, 0
for i, p in enumerate(pkls):
    tok = os.path.basename(os.path.dirname(p))
    try:
        try: mc = pickle.load(lzma.open(p, "rb"))
        except Exception: mc = pickle.load(open(p, "rb"))
        poses = np.asarray(mc.human_trajectory.poses, dtype=np.float32)
        n = poses.shape[0]
        sub[tok] = Trajectory(poses, TrajectorySampling(time_horizon=float(n)*0.5, interval_length=0.5))
    except Exception:
        bad += 1
    if (i+1) % 2000 == 0: print(f"  {i+1}/{len(pkls)}", flush=True)
print("built human-GT trajs:", len(sub), " bad:", bad, flush=True)
out = {"team_name":"youdrive","authors":"youdrive","email":"youdrive@local",
       "institution":"youdrive","country / region":"cn",
       "first_stage_predictions":[sub], "second_stage_predictions":[{}]}
pickle.dump(out, open(f"{ref}/submission.pkl", "wb"))
print("wrote", f"{ref}/submission.pkl", flush=True)
PY

echo "[calib] scoring HUMAN-GT submission (navtest one-stage EPDMS)..."
NAVSIM_DATA=/root/workspace/closed_loop/data/navsim bash "$SCORE_SH" "$REF/submission.pkl" navtest "$V2_CACHE"

echo "[calib] recomputing CORRECT EPDMS (excluding two_frame) from the CSV..."
"$NAVSIM_PY" - "$REF" <<'PY'
import glob, pandas as pd, numpy as np, sys
ref = sys.argv[1]
csv = sorted(glob.glob(f"{ref}/pdm_score_*/*.csv"))[-1]
df = pd.read_csv(csv)
per = df[~df["token"].astype(str).str.startswith("extended_pdm_score_")].copy()
v = per[per["valid"].astype(str).str.lower().eq("true")].copy()
g = lambda c: pd.to_numeric(v[c], errors="coerce")
W = dict(p=5.0, t=5.0, lk=2.0, hc=2.0); den = sum(W.values())
mult = g("multiplicative_metrics_prod_stage_one")
wavg = (W['p']*g("ego_progress_stage_one") + W['t']*g("time_to_collision_within_bound_stage_one")
        + W['lk']*g("lane_keeping_stage_one") + W['hc']*g("history_comfort_stage_one")) / den
correct = mult * wavg
print("\n==================== HUMAN-GT calibration (navtest one-stage) ====================")
print(f"  n valid            = {len(v)}")
print(f"  raw reported EPDMS  = {g('score').mean():.4f}   (buggy: two_frame=0 zeroes good tokens)")
print(f"  CORRECT EPDMS       = {correct.mean():.4f}   (excl two_frame)")
for c,lab in [("no_at_fault_collisions_stage_one","NC"),("drivable_area_compliance_stage_one","DAC"),
              ("driving_direction_compliance_stage_one","DDC"),("traffic_light_compliance_stage_one","TLC"),
              ("ego_progress_stage_one","EP"),("time_to_collision_within_bound_stage_one","TTC"),
              ("lane_keeping_stage_one","LK"),("history_comfort_stage_one","HC")]:
    if c in v.columns: print(f"    {lab:<4s} {g(c).mean():.4f}")
print("=================================================================================")
PY
echo "[calib] DONE."
