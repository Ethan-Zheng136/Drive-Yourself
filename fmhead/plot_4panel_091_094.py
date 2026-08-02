"""plot_4panel_091_094.py -- per-scene 2x2 four-panel comparison viz.

Layout (per scene, one PNG):
    row 0 (top)    = "my-style" ego-BEV trajectory plots (same style as plot_compare_091_094.py)
    row 1 (bottom) = navsim CLASSIC BEV render (drivable area / lanes / map + agents/boxes)
                     with the predicted ego trajectory (+ human GT) overlaid.
    col 0 (left)   = fm2 0.91 (GOLD full-FT v2 head, style OFF, alpha=0, pdm-select)
    col 1 (right)  = fm2 0.94 (style-adapter head, alpha=1, adapter ON, pdm-select)

Predicted trajectories are REUSED from the previous run's dumps (no inference):
    LEFT_JSON  (0.91)  {token: [[x_forward, y_left], ...] (10 cumulative ego-BEV pts)}
    RIGHT_JSON (0.94)  same format & tokens
Reference overlays (GT future / ego history) come from the navtest_nocot per-token json
for the top row, and from the loaded navsim Scene (get_future_trajectory) for the bottom row.

navsim classic BEV: navsim.visualization.bev.add_configured_bev_on_ax(ax, scene.map_api, frame)
renders the full map (lanes / walkways / intersections / crosswalks / baseline paths) + all
tracked-object boxes + the ego box. We then overlay the predicted ego trajectory (from the
dump, already in the ego-local x=forward/y=left frame navsim expects) via the SAME plotting
convention navsim uses (ax.plot(y_left, x_forward)), and the human GT via navsim's own
add_trajectory_to_bev_ax. Scenes are loaded once via navsim SceneLoader over the navtest logs.

Env:
  LEFT_JSON   (default /mnt/pfs/.../viz_091_vs_094/left/persona_a0.json)
  RIGHT_JSON  (default /mnt/pfs/.../viz_091_vs_094/right/persona_a1.json)
  TOKENS      (json list; default /root/workspace/fmhead/viz_tokens_240.json) -- first N used
  N_TOKENS    (default 200)
  OUTDIR      (default /mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_094)
  IMG_DIR     (default OUTDIR/images)
  JSON_DIR    (default AutoVLA/dataset/nuplan/navtest_nocot)  -- top-row GT/history overlays
  DPI         (default 140)   MONTAGE_N (default 12)
Data roots (navsim scene loading; must be exported before running, see run_viz_4panel.sh):
  OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, NAVSIM_EXP_ROOT
"""
import os
import json
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
PREV = "/mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094"
OUTDIR = os.environ.get("OUTDIR", "/mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_094")
LEFT_JSON = os.environ.get("LEFT_JSON", os.path.join(PREV, "left", "persona_a0.json"))
RIGHT_JSON = os.environ.get("RIGHT_JSON", os.path.join(PREV, "right", "persona_a1.json"))
TOKENS_JSON = os.environ.get("TOKENS", "/root/workspace/fmhead/viz_tokens_240.json")
N_TOKENS = int(os.environ.get("N_TOKENS", "200"))
IMG_DIR = os.environ.get("IMG_DIR", os.path.join(OUTDIR, "images"))
JSON_DIR = os.environ.get("JSON_DIR", os.path.join(REPO, "dataset/nuplan/navtest_nocot"))
DPI = int(os.environ.get("DPI", "140"))
MONTAGE_N = int(os.environ.get("MONTAGE_N", "12"))

LEFT_LABEL = os.environ.get("LEFT_LABEL", "fm2 0.91")
RIGHT_LABEL = os.environ.get("RIGHT_LABEL", "fm2 0.94")
LEFT_COLOR = "#1f77b4"    # blue
RIGHT_COLOR = "#d62728"   # red
GT_COLOR = "#9e9e9e"      # light gray
HIST_COLOR = "#c7c7c7"    # lighter gray


# ----------------------------------------------------------------------------
# Top-row "my-style" trajectory rendering (reused from plot_compare_091_094.py)
# ----------------------------------------------------------------------------
def to_screen(arr):
    """(...,>=2) [x_forward, y_left] -> screen (sx=-y_left, sy=x_forward)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    sx = -a[:, 1]
    sy = a[:, 0]
    return sx, sy


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
    ax.set_title(f"{label}  (my-style traj)", fontsize=11, fontweight="bold", color=color)
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


# ----------------------------------------------------------------------------
# Bottom-row navsim classic BEV rendering
# ----------------------------------------------------------------------------
def draw_navsim_panel(ax, scene, frame_idx, pred_xy, label, color, add_traj_fn, human_traj,
                      traj_cfg_human, want_map):
    """Render navsim classic BEV (map + agents + ego) and overlay predicted ego traj + human GT.

    pred_xy is [x_forward, y_left] cumulative in the ego-local frame. navsim BEV plots as
    ax.plot(y_left, x_forward) with an inverted x-axis (left = +y on the left), so we overlay
    with the same convention (prepend the ego origin (0,0), like add_trajectory_to_bev_ax).
    """
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_bev_ax, configure_ax

    simplified = False
    try:
        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    except Exception as e:  # map render failed -> fall back to agents/boxes only, note it
        simplified = True
        print(f"    [navsim] map render failed ({e!r}); agents/boxes-only fallback", flush=True)
        from navsim.visualization.bev import add_annotations_to_bev_ax
        add_annotations_to_bev_ax(ax, scene.frames[frame_idx].annotations)

    # human GT (navsim's own green human line) for reference
    if human_traj is not None:
        try:
            add_traj_fn(ax, human_traj, traj_cfg_human)
        except Exception as e:  # noqa: BLE001 - report, keep predicted overlay
            print(f"    [navsim] human traj overlay failed ({e!r})", flush=True)

    # predicted ego trajectory overlay (this model's prediction), colored to match top row
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
    ax.set_title(ttl, fontsize=11, fontweight="bold", color=color)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    return simplified


# ----------------------------------------------------------------------------
def build_scene_loader(tokens):
    from pathlib import Path
    from navsim.common.dataloader import SceneLoader, SceneFilter
    from navsim.common.dataclasses import SensorConfig
    data_split = os.environ.get("NAVSIM_DATA_SPLIT", "test")
    log_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / data_split
    sf = SceneFilter(
        num_history_frames=4, num_future_frames=10, frame_interval=1,
        has_route=True, tokens=list(tokens),
    )
    print(f"[4panel] building SceneLoader over {log_path} for {len(tokens)} tokens ...", flush=True)
    loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=None,
        scene_filter=sf,
        sensor_config=SensorConfig.build_no_sensors(),
    )
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

    # navsim imports & scene loading (once)
    from navsim.visualization.bev import add_trajectory_to_bev_ax
    from navsim.visualization.config import TRAJECTORY_CONFIG
    loader = build_scene_loader(tokens)
    scene_tokens = set(loader.tokens)
    frame_idx = 3  # num_history_frames - 1 (current frame)

    rendered, divs, no_scene, simplified_scenes = [], [], [], []
    for i, tok in enumerate(tokens):
        pL = left[tok]
        pR = right[tok]
        # top-row overlays from navtest_nocot json
        gt = hist = None
        jp = os.path.join(JSON_DIR, tok + ".json")
        if os.path.exists(jp):
            d = json.load(open(jp))
            gt = d.get("gt_trajectory")
            hist = d.get("his_trajectory")
        xlim, ylim = scene_limits(pL, pR, gt, hist)
        div = endpoint_div(pL, pR)
        divs.append((tok, div))

        # bottom-row navsim scene
        scene = human_traj = None
        if tok in scene_tokens:
            try:
                scene = loader.get_scene_from_token(tok)
                human_traj = scene.get_future_trajectory()
            except Exception as e:  # noqa: BLE001 - report, still emit top row
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
                ax.set_title(f"{lab}  (navsim classic BEV -- N/A)", fontsize=11,
                             fontweight="bold", color=colr)
                ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"navtest {tok}   |   endpoint div ({LEFT_LABEL} vs {RIGHT_LABEL}) = {div:.2f} m\n"
                     f"top: my-style ego-BEV traj   |   bottom: navsim classic BEV   "
                     f"(L={LEFT_LABEL} pdm-select  |  R={RIGHT_LABEL} pdm-select)",
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

    # montage: MONTAGE_N most-divergent scenes (most interesting)
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
                 f"(4-panel: my-style traj + navsim classic BEV)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    montage = os.path.join(OUTDIR, "montage.png")
    fig.savefig(montage, dpi=110)
    plt.close(fig)
    print(f"[4panel] montage -> {montage}")

    arr = np.array([d for _, d in divs_valid])
    if len(arr):
        print(f"[4panel] endpoint-div (m): mean={arr.mean():.2f} median={np.median(arr):.2f} "
              f"p90={np.percentile(arr,90):.2f} max={arr.max():.2f}")
    summary = {
        "n_images": len(rendered), "img_dir": IMG_DIR, "montage": montage,
        "n_requested": len(all_tokens), "n_with_both_preds": len(tokens),
        "n_no_navsim_scene": len(no_scene), "no_navsim_scene_tokens": no_scene,
        "n_simplified_navsim": len(simplified_scenes), "simplified_navsim_tokens": simplified_scenes,
        "left_ckpt_meta": lmeta, "right_ckpt_meta": rmeta,
        "endpoint_div_mean": float(arr.mean()) if len(arr) else None,
        "endpoint_div_median": float(np.median(arr)) if len(arr) else None,
        "top_divergent": pick,
        "left_attribution": os.environ.get(
            "LEFT_ATTR", "fm2 0.91 = GOLD full-FT v2 FMHead, style OFF (alpha=0), pdm-select"),
        "right_attribution": os.environ.get(
            "RIGHT_ATTR", "fm2 0.94 = style-adapter FMHead, alpha=1 (adapter ON), pdm-select"),
    }
    json.dump(summary, open(os.path.join(OUTDIR, "viz_summary.json"), "w"), indent=2)
    print(f"[4panel] summary -> {os.path.join(OUTDIR, 'viz_summary.json')}")


if __name__ == "__main__":
    main()
