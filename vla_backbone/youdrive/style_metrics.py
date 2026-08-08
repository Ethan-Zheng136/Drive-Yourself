"""YouDrive Style-Metrics engine (model-agnostic).

Runs AFTER inference: consumes a {token: trajectory} dump (the same format the
infer step emits) and computes context-conditional style metrics against the
human reference distribution in styletest.json.

Style frame (literature-grounded, Chalmers 76-driver FOT 2-factor result):
  F1  Assertiveness  = margin/decel/jerk  (small margin, hard late braking, jerky -> high)
  F2  Pace           = speed preference + lateral/lane-change activity

Each raw feature is mapped to a percentile within the SAME scenario_type bucket
of the human population (context conditionalization). Style position = mean
percentile over the features that load on each factor.

Usage:
  python style_metrics.py --dumps d1.json[,d2.json,...] \
      --styletest /root/workspace/tools/scorer/data/styledrive/styletest.json \
      --out metrics.json
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

# ---------------- feature extraction (identical definitions for human & model) ----------------

def _series_from_positions(xy, dt):
    """xy: list[[x_fwd, y_left]] cumulative from ego origin. Prepend (0,0) at t=0."""
    p = np.asarray([[0.0, 0.0]] + list(xy), dtype=float)
    d = np.diff(p, axis=0)
    vx = d[:, 0] / dt
    vy = d[:, 1] / dt
    return vx, vy

def _features(vx, vy, dt):
    """Kinematic features shared by human (from vx/vy series) and model (derived)."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    speed = np.hypot(vx, vy)
    ax = np.diff(speed) / dt if len(speed) > 1 else np.array([0.0])
    jerk = np.diff(ax) / dt if len(ax) > 1 else np.array([0.0])
    ay = np.diff(vy) / dt if len(vy) > 1 else np.array([0.0])
    return {
        "v_avg": float(np.mean(speed)),
        "v_max": float(np.max(speed)),
        "peak_decel": float(-np.min(ax)) if ax.size else 0.0,   # magnitude of hardest braking
        "peak_accel": float(np.max(ax)) if ax.size else 0.0,
        "jerk_rms": float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.0,
        "lat_vmax": float(np.max(np.abs(vy))) if vy.size else 0.0,
        "lat_amax": float(np.max(np.abs(ay))) if ay.size else 0.0,
    }

def human_features(rec):
    vx = rec.get("vx_ego"); vy = rec.get("vy_ego")
    if not vx or not vy:
        return None
    dt = 0.5
    f = _features(vx, vy, dt)
    mff = rec.get("min_front_frame") or [None, None]
    f["headway"] = float(mff[1]) if mff and mff[1] is not None else np.nan
    f["lane_change"] = 1.0 if rec.get("lane_change_frame", -1) != -1 else 0.0
    return f

def model_features(xy, dt):
    vx, vy = _series_from_positions(xy, dt)
    f = _features(vx, vy, dt)
    f["headway"] = np.nan          # lead distance needs scene context (v1); excluded from F1 here
    f["lane_change"] = np.nan      # derived lane-change detection (v1)
    return f

# F1 (assertiveness): higher percentile = more aggressive. headway is inverted (small gap = aggressive)
F1_POS = ["peak_decel", "peak_accel", "jerk_rms"]
F1_INV = ["headway"]
F2_POS = ["v_avg", "lat_vmax", "lane_change"]

# ---------------- human reference (context-conditional distributions) ----------------

def build_reference(styletest):
    """Returns {scenario_type: {feature: sorted np.array}} and a global fallback."""
    by_ctx, glob = {}, {}
    for tok, rec in styletest.items():
        f = human_features(rec)
        if f is None:
            continue
        ctx = rec.get("scenario_type", "unknown")
        by_ctx.setdefault(ctx, []).append(f)
        glob.setdefault("__all__", []).append(f)
    def _stack(lst):
        feats = {}
        keys = set().union(*[d.keys() for d in lst])
        for k in keys:
            vals = np.asarray([d[k] for d in lst if not (isinstance(d[k], float) and math.isnan(d[k]))], float)
            if vals.size:
                feats[k] = np.sort(vals)
        return feats
    ref = {ctx: _stack(lst) for ctx, lst in by_ctx.items()}
    ref["__all__"] = _stack(glob["__all__"])
    counts = {ctx: len(lst) for ctx, lst in by_ctx.items()}
    return ref, counts

def percentile_of(value, sorted_arr):
    if value is None or (isinstance(value, float) and math.isnan(value)) or sorted_arr is None or sorted_arr.size == 0:
        return np.nan
    return float(np.searchsorted(sorted_arr, value, side="right") / sorted_arr.size)

def style_coords(feat, ctx, ref):
    table = ref.get(ctx) or ref["__all__"]
    def comp(pos, inv):
        ps = []
        for k in pos:
            p = percentile_of(feat.get(k), table.get(k) if table else None)
            if not math.isnan(p): ps.append(p)
        for k in inv:
            p = percentile_of(feat.get(k), table.get(k) if table else None)
            if not math.isnan(p): ps.append(1.0 - p)
        return float(np.mean(ps)) if ps else np.nan
    return comp(F1_POS, F1_INV), comp(F2_POS, [])

# ---------------- elasticity (which scenarios can actually express style) ----------------

def elasticity(ref, counts, min_n=20):
    """Per-scenario style elasticity = mean spread (IQR) of human style-relevant features."""
    out = {}
    for ctx, table in ref.items():
        if ctx == "__all__" or counts.get(ctx, 0) < min_n:
            continue
        spreads = []
        for k in F1_POS + F2_POS:
            a = table.get(k)
            if a is not None and a.size > 4:
                iqr = np.subtract(*np.percentile(a, [75, 25]))
                rng = (a.max() - a.min()) or 1.0
                spreads.append(iqr / rng)
        if spreads:
            out[ctx] = {"elasticity": float(np.mean(spreads)), "n": counts[ctx]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["elasticity"]))

# ---------------- main scoring ----------------

def hull_area(pts):
    pts = np.asarray([p for p in pts if not any(math.isnan(c) for c in p)])
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)
    except Exception:
        return float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="comma-sep list of {token:traj} json dumps")
    ap.add_argument("--styletest", default="/root/workspace/tools/scorer/data/styledrive/styletest.json")
    ap.add_argument("--out", default="youdrive/style_metrics_out.json")
    args = ap.parse_args()

    styletest = json.load(open(args.styletest))
    ref, counts = build_reference(styletest)
    elas = elasticity(ref, counts)

    models = {}
    for path in args.dumps.split(","):
        path = path.strip()
        d = json.load(open(path))
        meta = d.get("_meta", {})
        name = meta.get("model", Path(path).stem)
        dt = float(meta.get("horizon_s", 4.0)) / float(meta.get("n_pts", 8))
        per = {}
        for tok, xy in d.items():
            if tok == "_meta":
                continue
            rec = styletest.get(tok, {})
            ctx = rec.get("scenario_type", "unknown")
            f = model_features(xy, dt)
            f1, f2 = style_coords(f, ctx, ref)
            per[tok] = {"ctx": ctx, "anc": rec.get("ANC_result"), "F1_assert": f1, "F2_pace": f2, "raw": f}
        models[name] = per

    # human GT style coords on the same tokens (reference baseline)
    eval_tokens = sorted({t for per in models.values() for t in per})
    human = {}
    for tok in eval_tokens:
        rec = styletest.get(tok)
        if not rec: continue
        f = human_features(rec)
        if f is None: continue
        f1, f2 = style_coords(f, rec.get("scenario_type", "unknown"), ref)
        human[tok] = {"F1_assert": f1, "F2_pace": f2, "anc": rec.get("ANC_result")}

    # aggregate metrics
    def coords(per): return [(v["F1_assert"], v["F2_pace"]) for v in per.values()]
    report = {"n_eval_tokens": len(eval_tokens), "models": {}, "human_ref": {}}
    for name, per in models.items():
        c = np.asarray([x for x in coords(per) if not any(math.isnan(v) for v in x)])
        report["models"][name] = {
            "F1_mean": float(np.nanmean([v["F1_assert"] for v in per.values()])),
            "F2_mean": float(np.nanmean([v["F2_pace"] for v in per.values()])),
            "coverage_hull": hull_area(coords(per)),
        }
    hc = [(v["F1_assert"], v["F2_pace"]) for v in human.values()]
    report["human_ref"] = {
        "F1_mean": float(np.nanmean([v["F1_assert"] for v in human.values()])) if human else None,
        "F2_mean": float(np.nanmean([v["F2_pace"] for v in human.values()])) if human else None,
        "coverage_hull": hull_area(hc),
    }

    # cross-model homogeneity = mean pairwise per-token style distance (quantifies visual overlap)
    names = list(models)
    if len(names) > 1:
        dists = []
        for tok in eval_tokens:
            pts = []
            for n in names:
                v = models[n].get(tok)
                if v and not math.isnan(v["F1_assert"]) and not math.isnan(v["F2_pace"]):
                    pts.append((v["F1_assert"], v["F2_pace"]))
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dists.append(math.dist(pts[i], pts[j]))
        report["cross_model_homogeneity"] = {
            "mean_pairwise_style_dist": float(np.mean(dists)) if dists else None,
            "note": "0 = identical styles (homogeneous); ~1.4 max in unit F1-F2 square",
        }

    report["top_elastic_scenarios"] = dict(list(elas.items())[:8])
    report["per_token"] = {n: per for n, per in models.items()}
    report["human_per_token"] = human

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    # ---- printed summary ----
    print("\n=== YouDrive Style-Metrics ===")
    print(f"eval tokens: {report['n_eval_tokens']}   human ref scenes: {sum(counts.values())}")
    print(f"\n{'model':<18}{'F1_assert':>11}{'F2_pace':>10}{'coverage':>11}")
    for n, m in report["models"].items():
        print(f"{n:<18}{m['F1_mean']:>11.3f}{m['F2_mean']:>10.3f}{m['coverage_hull']:>11.4f}")
    h = report["human_ref"]
    print(f"{'HUMAN-GT':<18}{(h['F1_mean'] or 0):>11.3f}{(h['F2_mean'] or 0):>10.3f}{h['coverage_hull']:>11.4f}")
    if "cross_model_homogeneity" in report:
        print(f"\ncross-model homogeneity (mean pairwise style dist): "
              f"{report['cross_model_homogeneity']['mean_pairwise_style_dist']:.4f}  (0=identical)")
    print("\ntop style-elastic scenario_types (where style is observable):")
    for ctx, v in report["top_elastic_scenarios"].items():
        print(f"  {ctx:<28} elasticity={v['elasticity']:.3f}  n={v['n']}")
    print(f"\nwritten: {args.out}")

if __name__ == "__main__":
    main()
