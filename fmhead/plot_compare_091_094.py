"""plot_compare_091_094.py -- per-scene side-by-side trajectory viz.

LEFT panel  = "fm2 0.91"  (GOLD full-FT v2 FMHead, PDMS 0.9088), predictions dumped by
              dump_fmhead_style.py into $LEFT_JSON  ({token: [[x,y],...], _meta:{...}}).
RIGHT panel = "fm2 0.94"  (style-adapter FMHead, navtest PDMS 0.9409), predictions in
              $RIGHT_JSON, same format & same tokens.

Reference overlays (light gray) come straight out of the navtest_nocot per-token json:
  gt_trajectory  (10 future poses, [x_forward, y_left, heading]) -> human/GT future
  his_trajectory (4 past poses)                                  -> ego history

Plot frame: ego BEV. Ego heading is UP. screen_x = -y_left (physical left -> screen left),
screen_y = x_forward. Equal aspect, shared axis limits per scene so the two panels are
directly comparable. Ego origin marked. High DPI.

Env:
  LEFT_JSON   (default OUTDIR/left/persona_a0.json)
  RIGHT_JSON  (default OUTDIR/right/persona_a1.json)
  OUTDIR      (default /mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094)
  IMG_DIR     (default OUTDIR/images)
  JSON_DIR    (default AutoVLA/dataset/nuplan/navtest_nocot)
  DPI         (default 140)   MONTAGE_N (default 12)
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
OUTDIR = os.environ.get("OUTDIR", "/mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094")
LEFT_JSON = os.environ.get("LEFT_JSON", os.path.join(OUTDIR, "left", "persona_a0.json"))
RIGHT_JSON = os.environ.get("RIGHT_JSON", os.path.join(OUTDIR, "right", "persona_a1.json"))
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


def to_screen(arr):
    """(...,>=2) [x_forward, y_left] -> screen (sx=-y_left, sy=x_forward)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    sx = -a[:, 1]
    sy = a[:, 0]
    return sx, sy


def draw_panel(ax, pred_xy, gt, hist, label, color, xlim, ylim):
    # history (past)
    if hist is not None and len(hist):
        hx, hy = to_screen(hist)
        ax.plot(hx, hy, "-", color=HIST_COLOR, lw=1.4, alpha=0.9, zorder=1)
        ax.scatter(hx, hy, s=8, color=HIST_COLOR, zorder=1)
    # GT future (human)
    if gt is not None and len(gt):
        gx, gy = to_screen(gt)
        ax.plot(gx, gy, "-", color=GT_COLOR, lw=2.2, alpha=0.95, zorder=2, label="GT / human")
        ax.scatter(gx[-1], gy[-1], s=26, color=GT_COLOR, zorder=2)
    # prediction
    if pred_xy is not None and len(pred_xy):
        px, py = to_screen(pred_xy)
        ax.plot(px, py, "-o", color=color, lw=2.4, ms=3.2, zorder=3, label=label)
        ax.scatter(px[-1], py[-1], s=40, color=color, edgecolors="white",
                   linewidths=0.8, zorder=4)
    # ego origin
    ax.scatter([0], [0], s=90, marker="*", color="black", zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(label, fontsize=12, fontweight="bold", color=color)
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
    """L2 distance between the two predictions' final xy (in metric x_forward,y_left)."""
    if a is None or b is None or not len(a) or not len(b):
        return float("nan")
    n = min(len(a), len(b))
    return float(np.linalg.norm(np.asarray(a[n - 1][:2]) - np.asarray(b[n - 1][:2])))


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    left = json.load(open(LEFT_JSON))
    right = json.load(open(RIGHT_JSON))
    lmeta = left.pop("_meta", {})
    rmeta = right.pop("_meta", {})
    print(f"[plot] LEFT  {LEFT_JSON}\n        meta={lmeta}")
    print(f"[plot] RIGHT {RIGHT_JSON}\n        meta={rmeta}")

    tokens = [t for t in left if t in right]
    print(f"[plot] common tokens: {len(tokens)}")

    rendered = []
    divs = []
    for tok in tokens:
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

        fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
        draw_panel(axes[0], pL, gt, hist, LEFT_LABEL, LEFT_COLOR, xlim, ylim)
        draw_panel(axes[1], pR, gt, hist, RIGHT_LABEL, RIGHT_COLOR, xlim, ylim)
        fig.suptitle(f"navtest {tok}   |   endpoint div = {div:.2f} m",
                     fontsize=11, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(IMG_DIR, f"{tok}.png")
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        rendered.append(tok)
        if len(rendered) % 25 == 0:
            print(f"[plot] {len(rendered)}/{len(tokens)} rendered", flush=True)

    print(f"[plot] DONE {len(rendered)} images -> {IMG_DIR}")

    # contact-sheet montage: pick the MONTAGE_N most-divergent scenes (most interesting)
    divs_valid = [(t, d) for t, d in divs if d == d]
    divs_valid.sort(key=lambda x: -x[1])
    pick = [t for t, _ in divs_valid[:MONTAGE_N]] or rendered[:MONTAGE_N]
    ncol = 3
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, tok in enumerate(pick):
        img = plt.imread(os.path.join(IMG_DIR, tok + ".png"))
        axes[i].imshow(img)
        axes[i].axis("off")
    fig.suptitle(f"{LEFT_LABEL} (L) vs {RIGHT_LABEL} (R) -- {len(pick)} most-divergent navtest scenes",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    montage = os.path.join(OUTDIR, "montage_top_divergent.png")
    fig.savefig(montage, dpi=110)
    plt.close(fig)
    print(f"[plot] montage -> {montage}")

    # summary stats for the report
    arr = np.array([d for _, d in divs_valid])
    if len(arr):
        print(f"[plot] endpoint-div (m): mean={arr.mean():.2f} median={np.median(arr):.2f} "
              f"p90={np.percentile(arr,90):.2f} max={arr.max():.2f}")
    json.dump({"n_images": len(rendered), "img_dir": IMG_DIR,
               "left_ckpt_meta": lmeta, "right_ckpt_meta": rmeta,
               "endpoint_div_mean": float(arr.mean()) if len(arr) else None,
               "endpoint_div_median": float(np.median(arr)) if len(arr) else None,
               "montage": montage, "top_divergent": pick},
              open(os.path.join(OUTDIR, "viz_summary.json"), "w"), indent=2)
    print(f"[plot] summary -> {os.path.join(OUTDIR, 'viz_summary.json')}")


if __name__ == "__main__":
    main()
