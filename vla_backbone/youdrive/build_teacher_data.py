"""Build a teacher-labeled SFT dataset: copy the base nocot JSONs and REPLACE gt_trajectory
with a teacher planner's trajectory, resampled to AutoVLA's 10-pose / 0.5s / 5.0s grid.

Teacher dumps are {token: [[x,y],...]} (typ. 8 pts @0.5s/4s) + a "_meta" with horizon_s / n_pts.
We interpolate onto 0.5..5.0s (10 pts); the last 1.0s (t>teacher horizon) is extrapolated at
constant velocity from the teacher's final segment (same choice used for the DDv2 teacher set).
Heading per pose = atan2(dy,dx) from the previous pose (origin for the first).

Reusable for any teacher (DDv2 / GoalFlow / TransFuser / ...). Output goes to PFS.

Env:
  NOCOT_DIR     base nocot JSON dir (e.g. /mnt/pfs/.../persona/nocot_navtrain12k)
  TEACHER_JSON  teacher dump (e.g. /mnt/pfs/.../persona/goalflow_navtrain12k.json)
  OUT_DIR       output dir on PFS (e.g. /mnt/pfs/.../persona/nocot_goalflow_teacher12k)
"""
import os, json, glob
import numpy as np

NOCOT_DIR = os.environ["NOCOT_DIR"]
TEACHER_JSON = os.environ["TEACHER_JSON"]
OUT_DIR = os.environ["OUT_DIR"]
assert OUT_DIR.startswith("/mnt/pfs/"), f"OUT_DIR must be on PFS, got {OUT_DIR}"
os.makedirs(OUT_DIR, exist_ok=True)

teacher = json.load(open(TEACHER_JSON))
meta = teacher.get("_meta", {})
H = float(meta.get("horizon_s", 4.0))
n = int(meta.get("n_pts", 8))
t_src = np.linspace(H / n, H, n)            # teacher grid, e.g. 0.5..4.0 (8 pts)
t_dst = np.linspace(0.5, 5.0, 10)           # AutoVLA grid: 10 poses @0.5s, 5.0s


def resample(xy):
    xy = np.asarray(xy, dtype=float)        # (n, 2)
    if xy.shape[0] < 2:
        raise ValueError(f"teacher traj too short: {xy.shape}")
    v_last = (xy[-1] - xy[-2]) / (t_src[-1] - t_src[-2])   # const-velocity tail
    out = np.zeros((10, 2))
    for i, t in enumerate(t_dst):
        if t <= t_src[-1] + 1e-6:
            out[i, 0] = np.interp(t, t_src, xy[:, 0])
            out[i, 1] = np.interp(t, t_src, xy[:, 1])
        else:
            out[i] = xy[-1] + v_last * (t - t_src[-1])
    return out


def headings(xy10):
    prev = np.array([0.0, 0.0])
    hs = []
    for p in xy10:
        d = p - prev
        hs.append(float(np.arctan2(d[1], d[0])))
        prev = p
    return hs


cnt, skipped = 0, 0
for f in glob.glob(os.path.join(NOCOT_DIR, "*.json")):
    d = json.load(open(f))
    tok = d.get("token") or os.path.splitext(os.path.basename(f))[0]
    if tok not in teacher:
        skipped += 1
        continue
    xy10 = resample(teacher[tok])
    hs = headings(xy10)
    d["gt_trajectory"] = [[float(xy10[i, 0]), float(xy10[i, 1]), hs[i]] for i in range(10)]
    json.dump(d, open(os.path.join(OUT_DIR, os.path.basename(f)), "w"))
    cnt += 1

print(f"[build_teacher_data] wrote {cnt} (skipped {skipped} not in teacher) -> {OUT_DIR}", flush=True)
