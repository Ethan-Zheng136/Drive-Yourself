"""Stage-2 style cache (precompute once, CPU).

For each navtrain token: DDv2's full kinematic+social profile (the style TARGET) +
cached other-agent futures (so the reward can compute the rollout's social features by
pure geometry, no scene reload) + per-feature normalization stats.

Run (with NUPLAN env vars set, like metric caching):
  OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim_train \
  python youdrive/build_style_cache.py
"""
import os, sys, json, math, pickle
import numpy as np
from pathlib import Path
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "navsim"))
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

DATA = os.environ.get("OPENSCENE_DATA_ROOT", "/root/workspace/closed_loop/data/navsim_train")
DDV2_JSON = os.environ.get("DDV2_JSON", "/mnt/pfs/zhengguantian/autovla/persona/ddv2_navtrain12k.json")
OUT = os.environ.get("OUT", "/mnt/pfs/zhengguantian/autovla/persona/style_cache_navtrain12k.pkl")
DT = 0.5
CORRIDOR = 1.5
VRU_KEYS = ("pedestrian", "bicycle", "cyclist", "bike")
KIN = ["v_avg", "v_std", "peak_acc", "peak_dec", "long_jerk", "lat_amax", "lat_jerk"]
SOC = ["thw_min", "ttc_min", "lead_gap_min", "dist_any_min", "dist_vru_min"]


def agents_in_t0(scene):
    nh = scene.scene_metadata.num_history_frames
    poses = np.asarray(scene.get_future_trajectory().poses)
    out = []
    for k in range(len(poses)):
        fidx = nh + k
        if fidx >= len(scene.frames):
            break
        ann = scene.frames[fidx].annotations
        boxes = np.asarray(ann.boxes, dtype=float)
        if boxes.size == 0:
            out.append((np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), bool))); continue
        loc = boxes[:, :2]
        x, y, th = poses[k]; c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        xy_t0 = (loc @ R.T) + np.array([x, y])
        vel = np.asarray(ann.velocity_3d, dtype=float)[:, :2] if np.asarray(ann.velocity_3d).size else np.zeros((len(loc), 2))
        is_vru = np.array([any(kw in (nm or "").lower() for kw in VRU_KEYS) for nm in ann.names])
        out.append((xy_t0, vel, is_vru))
    return out


def ego_series(xy):
    p = np.array([[0.0, 0.0]] + list(xy), dtype=float)
    v = np.diff(p, axis=0) / DT
    return p[1:], np.hypot(v[:, 0], v[:, 1])


def interact(ego_xy, agents):
    pos, sp = ego_series(ego_xy)
    K = min(len(pos), len(agents))
    thw = ttc = lead_gap = math.inf; dmin = dvru = math.inf
    for k in range(K):
        exy = pos[k]; espeed = max(sp[k], 1e-3)
        axy, avel, vru = agents[k]
        if len(axy) == 0:
            continue
        d = np.hypot(axy[:, 0] - exy[0], axy[:, 1] - exy[1])
        dmin = min(dmin, float(d.min()))
        if vru.any():
            dvru = min(dvru, float(d[vru].min()))
        ahead = (axy[:, 0] > exy[0]) & (np.abs(axy[:, 1] - exy[1]) < CORRIDOR)
        if ahead.any():
            gap = axy[ahead, 0] - exy[0]
            j = int(np.argmin(gap)); g = float(gap[j])
            lead_gap = min(lead_gap, g); thw = min(thw, g / espeed)
            close = espeed - float(avel[ahead][j, 0])
            if close > 0.1:
                ttc = min(ttc, g / close)
    f = lambda v: (None if math.isinf(v) else round(v, 3))
    return {"thw_min": f(thw), "ttc_min": f(ttc), "lead_gap_min": f(lead_gap),
            "dist_any_min": f(dmin), "dist_vru_min": f(dvru)}


def kin(xy):
    p = np.array([[0.0, 0.0]] + list(xy), float); v = np.diff(p, axis=0) / DT
    vx, vy = v[:, 0], v[:, 1]; sp = np.hypot(vx, vy)
    ax = np.diff(sp) / DT if len(sp) > 1 else np.array([0.])
    lj = np.diff(ax) / DT if len(ax) > 1 else np.array([0.])
    ay = np.diff(vy) / DT if len(vy) > 1 else np.array([0.])
    aj = np.diff(ay) / DT if len(ay) > 1 else np.array([0.])
    return {"v_avg": float(sp.mean()), "v_std": float(sp.std()),
            "peak_acc": float(ax.max()) if ax.size else 0.0, "peak_dec": float(-ax.min()) if ax.size else 0.0,
            "long_jerk": float(np.sqrt((lj ** 2).mean())) if lj.size else 0.0,
            "lat_amax": float(np.abs(ay).max()) if ay.size else 0.0,
            "lat_jerk": float(np.sqrt((aj ** 2).mean())) if aj.size else 0.0}


def resample_teacher(xy, meta):
    """Resample a teacher trajectory onto the canonical 8-pt @0.5s (0.5..4.0s) grid so jerk/
    accel are on the SAME time base as DDv2 and the student rollout. Teacher dumps differ:
    DDv2/WoTE=8@0.5s, GTRS/Hydra=40@0.1s. Without this, the fixed DT=0.5 kin() mis-scales the
    40-pt dumps (~25-350x) and the style reward collapses to ~0."""
    xy = np.asarray(xy, float)[:, :2]
    H = float(meta.get("horizon_s", 4.0)); n = int(meta.get("n_pts", len(xy)))
    t_dst = np.arange(DT, H + 1e-6, DT)                       # 0.5 .. H (8 pts @ H=4.0)
    if n == len(t_dst) and abs(H / max(n, 1) - DT) < 1e-6:
        return xy                                            # already on the 0.5s grid
    t_src = np.concatenate([[0.0], np.linspace(H / n, H, n)[:len(xy)]])
    xy0 = np.concatenate([[[0.0, 0.0]], xy[:n]], axis=0)
    return np.stack([np.interp(t_dst, t_src, xy0[:, 0]),
                     np.interp(t_dst, t_src, xy0[:, 1])], axis=1)


def main():
    ddv2 = json.load(open(DDV2_JSON))
    META = ddv2.get("_meta", {})
    print(f"[build_style_cache] teacher _meta={META}", flush=True)
    toks = [t for t in ddv2 if t != "_meta"]
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                     has_route=True, max_scenes=None, log_names=None, tokens=toks)
    loader = SceneLoader(data_path=Path(f"{DATA}/navsim_logs/trainval"),
                         sensor_blobs_path=Path(f"{DATA}/sensor_blobs/trainval"),
                         scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    common = [t for t in loader.tokens if t in ddv2]
    print("tokens to process:", len(common), flush=True)
    cache = {}
    for i, tok in enumerate(common):
        if i % 500 == 0:
            print(f"  {i}/{len(common)}", flush=True)
        try:
            sc = loader.get_scene_from_token(tok)
            ag = agents_in_t0(sc)
            dt = resample_teacher(ddv2[tok], META)
            prof = {**kin(dt), **interact(dt, ag)}
            cache[tok] = {"profile": prof,
                          "agents": [(a.tolist(), v.tolist(), r.tolist()) for a, v, r in ag]}
        except Exception as e:
            print("skip", tok, repr(e)[:80])
    # per-feature normalization stats (from DDv2 distribution; social Nones skipped)
    feats = KIN + SOC
    vals = {f: [] for f in feats}
    for d in cache.values():
        for f in feats:
            x = d["profile"].get(f)
            if x is not None:
                vals[f].append(x)
    norm = {f: (float(np.mean(vals[f])), float(np.std(vals[f]) + 1e-6)) for f in feats if vals[f]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump({"cache": cache, "norm": norm, "feats": feats}, open(OUT, "wb"))
    print(f"saved {OUT}  tokens={len(cache)}", flush=True)
    print("norm stats:", {k: (round(m, 2), round(s, 2)) for k, (m, s) in norm.items()})


if __name__ == "__main__":
    main()
