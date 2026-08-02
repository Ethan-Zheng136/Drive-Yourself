"""Score a raw teacher trajectory dump (e.g. goalflow_navtrain12k_fixed.json) for
native PDMS on a metric cache. Same resample (8->10) + heading + rl_pdm_score
logic as /tmp/teacher_native_pdms.py, but reads the {token: [[x,y],...]} dump
directly (no nocot dir needed).

Env:
  TEACHER_JSON  teacher dump json  (default goalflow_navtrain12k_fixed.json)
  METRIC_CACHE  metric cache dir   (default metric_cache_navtrain12k)
  N_SAMPLE      random tokens to score (default 300; 0 = all)
Usage (autovla env, from AutoVLA repo root):
  TEACHER_JSON=/mnt/pfs/zhengguantian/autovla/persona/goalflow_navtrain12k_fixed.json \
  /root/workspace/miniconda3/envs/autovla/bin/python youdrive/score_teacher_dump_pdms.py
"""
import os, sys, json, random
from pathlib import Path
import numpy as np

AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
sys.path.insert(0, AV); sys.path.insert(0, f"{AV}/navsim")
from models.utils.score import PDM_Reward, Trajectory, TrajectorySampling

TEACHER_JSON = os.environ.get("TEACHER_JSON",
    "/mnt/pfs/zhengguantian/autovla/persona/goalflow_navtrain12k_fixed.json")
MC = Path(os.environ.get("METRIC_CACHE",
    "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtrain12k"))
N_SAMPLE = int(os.environ.get("N_SAMPLE", "300"))

teacher = json.load(open(TEACHER_JSON))
meta = teacher.get("_meta", {})
H = float(meta.get("horizon_s", 4.0)); n = int(meta.get("n_pts", 8))
t_src = np.linspace(H / n, H, n)
t_dst = np.linspace(0.5, 5.0, 10)

def resample(xy):
    xy = np.asarray(xy, float)
    v = (xy[-1] - xy[-2]) / (t_src[-1] - t_src[-2])
    out = np.zeros((10, 2))
    for i, t in enumerate(t_dst):
        if t <= t_src[-1] + 1e-6:
            out[i, 0] = np.interp(t, t_src, xy[:, 0]); out[i, 1] = np.interp(t, t_src, xy[:, 1])
        else:
            out[i] = xy[-1] + v * (t - t_src[-1])
    return out

def headings(xy10):
    prev = np.array([0.0, 0.0]); hs = []
    for p in xy10:
        d = p - prev; hs.append(np.arctan2(d[1], d[0])); prev = p
    return np.array(hs)

rew = PDM_Reward(MC)
avail = set(rew.metric_cache_loader.metric_cache_paths.keys())
samp = TrajectorySampling(num_poses=10, interval_length=0.5)
toks = sorted((set(teacher) - {"_meta"}) & avail)
random.seed(0); random.shuffle(toks)
if N_SAMPLE > 0:
    toks = toks[:N_SAMPLE]
print(f"scoring {len(toks)} tokens from {TEACHER_JSON}")

scores = []
for tk in toks:
    try:
        xy10 = resample(teacher[tk]); hs = headings(xy10)
        traj = Trajectory(np.concatenate([xy10, hs[:, None]], 1), samp)
        s = rew.rl_pdm_score(traj, tk)
        if s is not None:
            scores.append(float(s))
    except Exception:
        pass
arr = np.array(scores)
print(f"GoalFlow(fixed) native PDMS = {arr.mean():.4f}  (n={len(arr)}, std={arr.std():.3f})")
