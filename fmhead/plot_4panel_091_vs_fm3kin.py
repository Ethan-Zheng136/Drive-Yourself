"""plot_4panel_091_vs_fm3kin.py -- per-scene 2x2 four-panel comparison viz for the
kinematic-feasibility story: fm2 0.91 (GOLD, LEFT) vs fm3-kin ep6 (PDMS 0.915, RIGHT).

Reuses the exact rendering of plot_4panel_091_094.py (top = my-style ego-BEV trajectory,
bottom = navsim classic BEV with full map + agents), only the labels / captions / default
paths differ. LEFT reuses the existing 0.91 dump; RIGHT reads the fm3-kin ep6 kinematic
dump (dump_fm3kin_viz.py). Per-panel we annotate the max longitudinal jerk (m/s^3) so the
"no acute kinks / low jerk" improvement is legible at a glance.

Env:
  LEFT_JSON   (default viz_091_vs_094/left/persona_a0.json)
  RIGHT_JSON  (default viz_4panel_091_vs_fm3kin_ep6/right/fm3kin_ep6.json)
  TOKENS      (json list; default viz_tokens_240.json) -- first N_TOKENS used
  N_TOKENS    (default 200)
  OUTDIR      (default viz_4panel_091_vs_fm3kin_ep6)   IMG_DIR (default OUTDIR/images)
  JSON_DIR    (default AutoVLA/dataset/nuplan/navtest_nocot)  DPI (140)  MONTAGE_N (12)
Data roots (navsim scene load) exported by run_viz_4panel_fm3kin.sh.
"""
import os
import json
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
OUTDIR = os.environ.get("OUTDIR", "/mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_vs_fm3kin_ep6")
LEFT_JSON = os.environ.get("LEFT_JSON", "/mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094/left/persona_a0.json")
RIGHT_JSON = os.environ.get("RIGHT_JSON", os.path.join(OUTDIR, "right", "fm3kin_ep6.json"))
TOKENS_JSON = os.environ.get("TOKENS", "/root/workspace/fmhead/viz_tokens_240.json")
N_TOKENS = int(os.environ.get("N_TOKENS", "200"))
IMG_DIR = os.environ.get("IMG_DIR", os.path.join(OUTDIR, "images"))
JSON_DIR = os.environ.get("JSON_DIR", os.path.join(REPO, "dataset/nuplan/navtest_nocot"))
DPI = int(os.environ.get("DPI", "140"))
MONTAGE_N = int(os.environ.get("MONTAGE_N", "12"))

LEFT_LABEL = os.environ.get("LEFT_LABEL", "fm2 0.91")
RIGHT_LABEL = os.environ.get("RIGHT_LABEL", "fm3-kin ep6 (0.915)")
DT = float(os.environ.get("JERK_DT", "0.5"))
LEFT_COLOR = "#1f77b4"    # blue
RIGHT_COLOR = "#2ca02c"   # green (fm3-kin: the "improved / feasible" side)
GT_COLOR = "#9e9e9e"
HIST_COLOR = "#c7c7c7"


def to_screen(arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return -a[:, 1], a[:, 0]


def max_jerk(xy, dt=DT):
    """Max |3rd finite diff| of the cumulative path (m/s^3) -- kink/jerk proxy."""
    p = np.asarray(xy, dtype=float)
    if p.shape[0] < 4:
        return float("nan")
    j = np.diff(np.diff(np.diff(p, axis=0), axis=0), axis=0) / (dt ** 3)
    return float(np.linalg.norm(j, axis=-1).max())


def draw_mystyle_panel(ax, pred_xy, gt, hist, label, color, xlim, ylim):
    if hist is not None and len(hist):
        hx, hy = to_screen(hist)
        ax.plot(hx, hy, "-", color=HIST_COLOR, lw=1.4, alpha=0.9, zorder=1)
        ax.scatter(hx, hy, s=8, color=HIST_COLOR, zorder=1)
    if gt is not None and len(gt):
        gx, gy = to_screen(gt)
        ax.plot(gx, gy, "-", color=GT_COLOR, lw=2.2, alpha=0.95, zorder=2, label="GT / human")
        ax.scatter(gx[-1], gy[-1], s=26, color=GT_COLOR, zorder=2)
    if pred_xy is not None and len(pred_xy):
        px, py = to_screen(pred_xy)
        ax.plot(px, py, "-o", color=color, lw=2.4, ms=3.2, zorder=3, label=label)
        ax.scatter(px[-1], py[-1], s=40, color=color, edgecolors="white",
                   linewidths=0.8, zorder=4)
    ax.scatter([0], [0], s=90, marker="*", color="black", zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.4)
    jk = max_jerk(pred_xy)
    ax.set_title(f"{label}  (my-style traj)   max-jerk={jk:.1f} m/s\u00b3",
                 fontsize=10.5, fontweight="bold", color=color)
    ax.set_xlabel("left (m)  [<- right | left ->]", fontsize=8)
    ax.set_ylabel("forward (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def scene_limits(*trajs, pad=3.0, min_half=8.0):
    xs, ys = [0.0], [0.0]
    for t in trajs:
        if t is None or not len(t):
            continue
        sx, sy = to_screen(t)
        xs.extend(sx.tolist())
        ys.extend(sy.tolist())
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max((xmax - xmin) / 2, (ymax - ymin) / 2, min_half) + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def endpoint_div(a, b):
    if a is None or b is None or not len(a) or not len(b):
        return float("nan")
    n = min(len(a), len(b))
    return float(np.linalg.norm(np.asarray(a[n - 1][:2]) - np.asarray(b[n - 1][:2])))


def draw_navsim_panel(ax, scene, frame_idx, pred_xy, label, color, add_traj_fn, human_traj,
                      traj_cfg_human, want_map):
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_bev_ax, configure_ax

    simplified = False
    try:
        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    except Exception as e:
        simplified = True
        print(f"    [navsim] map render failed ({e!r}); agents/boxes-only fallback", flush=True)
        from navsim.visualization.bev import add_annotations_to_bev_ax
        add_annotations_to_bev_ax(ax, scene.frames[frame_idx].annotations)

    if human_traj is not None:
        try:
            add_traj_fn(ax, human_traj, traj_cfg_human)
        except Exception as e:  # noqa: BLE001
            print(f"    [navsim] human traj overlay failed ({e!r})", flush=True)

    if pred_xy is not None and len(pred_xy):
        p = np.asarray(pred_xy, dtype=float)
        poses = np.concatenate([np.array([[0.0, 0.0]]), p[:, :2]])
        ax.plot(poses[:, 1], poses[:, 0], "-o", color=color, lw=2.4, ms=3.2,
                markeredgecolor="white", zorder=4, label=f"{label} pred")
        ax.scatter([poses[-1, 1]], [poses[-1, 0]], s=45, color=color,
                   edgecolors="white", linewidths=0.9, zorder=5)

    configure_bev_ax(ax)
    configure_ax(ax)
    ttl = f"{label}  (navsim classic BEV{'' if want_map and not simplified else ' -- agents only'})"
    ax.set_title(ttl, fontsize=10.5, fontweight="bold", color=color)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    return simplified


def build_scene_loader(tokens):
    from pathlib import Path
    from navsim.common.dataloader import SceneLoader, SceneFilter
    from navsim.common.dataclasses import SensorConfig
    data_split = os.environ.get("NAVSIM_DATA_SPLIT", "test")
    log_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / data_split
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                     has_route=True, tokens=list(tokens))
    print(f"[4panel] building SceneLoader over {log_path} for {len(tokens)} tokens ...", flush=True)
    loader = SceneLoader(data_path=log_path, sensor_blobs_path=None, scene_filter=sf,
                         sensor_config=SensorConfig.build_no_sensors())
    print(f"[4panel] SceneLoader ready: {len(loader.tokens)} scenes available", flush=True)
    return loader


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    left = json.load(open(LEFT_JSON))
    right = json.load(open(RIGHT_JSON))
    lmeta = left.pop("_meta", {})
    rmeta = right.pop("_meta", {})
    print(f"[4panel] LEFT  {LEFT_JSON}\n         meta={lmeta}")
    print(f"[4panel] RIGHT {RIGHT_JSON}\n         meta={rmeta}")

    all_tokens = json.load(open(TOKENS_JSON))[:N_TOKENS]
    tokens = [t for t in all_tokens if t in left and t in right]
    print(f"[4panel] requested {len(all_tokens)} tokens; {len(tokens)} have both predictions")

    from navsim.visualization.bev import add_trajectory_to_bev_ax
    from navsim.visualization.config import TRAJECTORY_CONFIG
    loader = build_scene_loader(tokens)
    scene_tokens = set(loader.tokens)
    frame_idx = 3

    rendered, divs, no_scene, simplified_scenes = [], [], [], []
    jerk_l, jerk_r = [], []
    for i, tok in enumerate(tokens):
        pL = left[tok]
        pR = right[tok]
        gt = hist = None
        jp = os.path.join(JSON_DIR, tok + ".json")
        if os.path.exists(jp):
            d = json.load(open(jp))
            gt = d.get("gt_trajectory")
            hist = d.get("his_trajectory")
        xlim, ylim = scene_limits(pL, pR, gt, hist)
        div = endpoint_div(pL, pR)
        divs.append((tok, div))
        jl, jr = max_jerk(pL), max_jerk(pR)
        jerk_l.append(jl); jerk_r.append(jr)

        scene = human_traj = None
        if tok in scene_tokens:
            try:
                scene = loader.get_scene_from_token(tok)
                human_traj = scene.get_future_trajectory()
            except Exception as e:  # noqa: BLE001
                print(f"  [scene] {tok} load failed: {e!r}", flush=True)
                traceback.print_exc()
                scene = None
        if scene is None:
            no_scene.append(tok)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.5))
        draw_mystyle_panel(axes[0, 0], pL, gt, hist, LEFT_LABEL, LEFT_COLOR, xlim, ylim)
        draw_mystyle_panel(axes[0, 1], pR, gt, hist, RIGHT_LABEL, RIGHT_COLOR, xlim, ylim)

        for col, (pred, lab, colr) in enumerate(
                [(pL, LEFT_LABEL, LEFT_COLOR), (pR, RIGHT_LABEL, RIGHT_COLOR)]):
            ax = axes[1, col]
            if scene is not None:
                simp = draw_navsim_panel(
                    ax, scene, frame_idx, pred, lab, colr,
                    add_trajectory_to_bev_ax, human_traj, TRAJECTORY_CONFIG["human"],
                    want_map=True)
                if simp and tok not in simplified_scenes:
                    simplified_scenes.append(tok)
            else:
                ax.text(0.5, 0.5, "navsim scene unavailable", ha="center", va="center",
                        fontsize=11, color="gray", transform=ax.transAxes)
                ax.set_title(f"{lab}  (navsim classic BEV -- N/A)", fontsize=10.5,
                             fontweight="bold", color=colr)
                ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            f"navtest {tok}   |   endpoint div (0.91 vs fm3-kin) = {div:.2f} m   "
            f"|   max-jerk  L={jl:.1f}  R={jr:.1f} m/s\u00b3\n"
            f"top: my-style ego-BEV traj   |   bottom: navsim classic BEV   "
            f"(L={LEFT_LABEL} GOLD xy-decode  |  R={RIGHT_LABEL} kinematic-unicycle decode, pdm-select)",
            fontsize=11, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        out = os.path.join(IMG_DIR, f"{tok}.png")
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        rendered.append(tok)
        if len(rendered) % 10 == 0:
            print(f"[4panel] {len(rendered)}/{len(tokens)} rendered "
                  f"(no_scene={len(no_scene)}, simplified={len(simplified_scenes)})", flush=True)

    print(f"[4panel] DONE {len(rendered)} images -> {IMG_DIR}")
    print(f"[4panel] scenes without navsim render: {len(no_scene)}")
    print(f"[4panel] scenes with simplified (agents-only) navsim panels: {len(simplified_scenes)}")

    divs_valid = [(t, d) for t, d in divs if d == d and t in rendered]
    divs_valid.sort(key=lambda x: -x[1])
    pick = [t for t, _ in divs_valid[:MONTAGE_N]] or rendered[:MONTAGE_N]
    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.0, nrow * 5.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, tok in enumerate(pick):
        img = plt.imread(os.path.join(IMG_DIR, tok + ".png"))
        axes[i].imshow(img)
        axes[i].axis("off")
    fig.suptitle(f"{LEFT_LABEL} vs {RIGHT_LABEL} -- {len(pick)} most-divergent navtest scenes "
                 f"(kinematic-feasibility: acute kinks/hooks in 0.91 vs smooth fm3-kin)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    montage = os.path.join(OUTDIR, "montage.png")
    fig.savefig(montage, dpi=110)
    plt.close(fig)
    print(f"[4panel] montage -> {montage}")

    arr = np.array([d for _, d in divs_valid])
    jl_arr = np.array([j for j in jerk_l if j == j])
    jr_arr = np.array([j for j in jerk_r if j == j])
    if len(arr):
        print(f"[4panel] endpoint-div (m): mean={arr.mean():.2f} median={np.median(arr):.2f} "
              f"p90={np.percentile(arr,90):.2f} max={arr.max():.2f}")
    if len(jl_arr) and len(jr_arr):
        print(f"[4panel] max-jerk (m/s^3): LEFT 0.91  mean={jl_arr.mean():.1f} p90={np.percentile(jl_arr,90):.1f} "
              f"max={jl_arr.max():.1f}")
        print(f"[4panel] max-jerk (m/s^3): RIGHT fm3-kin mean={jr_arr.mean():.1f} p90={np.percentile(jr_arr,90):.1f} "
              f"max={jr_arr.max():.1f}")
    summary = {
        "n_images": len(rendered), "img_dir": IMG_DIR, "montage": montage,
        "n_requested": len(all_tokens), "n_with_both_preds": len(tokens),
        "n_no_navsim_scene": len(no_scene), "no_navsim_scene_tokens": no_scene,
        "n_simplified_navsim": len(simplified_scenes), "simplified_navsim_tokens": simplified_scenes,
        "left_ckpt_meta": lmeta, "right_ckpt_meta": rmeta,
        "endpoint_div_mean": float(arr.mean()) if len(arr) else None,
        "endpoint_div_median": float(np.median(arr)) if len(arr) else None,
        "jerk_left_mean": float(jl_arr.mean()) if len(jl_arr) else None,
        "jerk_left_p90": float(np.percentile(jl_arr, 90)) if len(jl_arr) else None,
        "jerk_left_max": float(jl_arr.max()) if len(jl_arr) else None,
        "jerk_right_mean": float(jr_arr.mean()) if len(jr_arr) else None,
        "jerk_right_p90": float(np.percentile(jr_arr, 90)) if len(jr_arr) else None,
        "jerk_right_max": float(jr_arr.max()) if len(jr_arr) else None,
        "top_divergent": pick,
        "left_attribution": "fm2 0.91 = GOLD full-FT v2 FMHead, xy-decode, style OFF, pdm-select",
        "right_attribution": "fm3-kin ep6 (PDMS 0.915) = kinematic-unicycle decode (accel<=5, yawrate<=1, v<=25), style OFF, pdm-select, v0 from ego velocity",
    }
    json.dump(summary, open(os.path.join(OUTDIR, "viz_summary.json"), "w"), indent=2)
    print(f"[4panel] summary -> {os.path.join(OUTDIR, 'viz_summary.json')}")


if __name__ == "__main__":
    main()
