"""Render StyleDrive A/N/C example scenes (BEV map + agents + ego future trajectory).
Same scenario_type (protected_intersections) so style differences are visible.
"""
import os, json, tempfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax

DATA = "/root/workspace/closed_loop/data/navsim"
# tokens picked from styledrive protected_intersections (token, v_avg)
PICK = {
    "A(aggressive)": [("e8e5d67c60ef5771", 12.5), ("772411741fee555d", 6.4), ("172823ca4ea3514f", 5.0)],
    "N(neutral)":    [("94af4752a875550e", 3.8), ("5509d617c7cc513e", 2.2), ("8151351c964a5c93", 4.6)],
    "C(conservative)":[("5caaa45d037a5773", 0.2), ("515fbde824af577c", 0.8), ("2ce2e9b16dec5c3b", 2.2)],
}
all_tokens = [t for v in PICK.values() for t, _ in v]

sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                 has_route=True, max_scenes=None, log_names=None, tokens=all_tokens)
loader = SceneLoader(
    data_path=Path(f"{DATA}/navsim_logs/trainval"),
    sensor_blobs_path=Path(f"{DATA}/sensor_blobs/trainval"),
    scene_filter=sf,
    sensor_config=SensorConfig.build_no_sensors(),
)
print("loaded tokens:", len(loader.tokens))

TRAJ = dict(line_color="#e6194B", line_color_alpha=1.0, line_width=2.5, line_style="-",
            marker="o", marker_size=3, marker_edge_color="black", zorder=10)

tmp = tempfile.mkdtemp()
pngs = {}
for style, items in PICK.items():
    for tok, v in items:
        try:
            scene = loader.get_scene_from_token(tok)
            fidx = scene.scene_metadata.num_history_frames - 1
            fig, ax = plot_bev_frame(scene, fidx)
            traj = scene.get_future_trajectory()
            add_trajectory_to_bev_ax(ax, traj, TRAJ)
            ax.set_title(f"{style}  v_avg={v} m/s", fontsize=11)
            p = f"{tmp}/{tok}.png"
            fig.savefig(p, dpi=90, bbox_inches="tight"); plt.close(fig)
            pngs[tok] = p
        except Exception as e:
            print(f"  skip {tok}: {e!r}")

# composite 3x3
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
for r, (style, items) in enumerate(PICK.items()):
    for c, (tok, v) in enumerate(items):
        ax = axes[r][c]; ax.axis("off")
        if tok in pngs:
            ax.imshow(mpimg.imread(pngs[tok]))
            ax.set_title(f"{style}  v_avg={v}", fontsize=12, fontweight="bold")
out = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/styledrive_scenes.png"
plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
