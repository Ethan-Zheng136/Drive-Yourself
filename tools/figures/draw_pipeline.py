import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path
import matplotlib.patches as mpatches

C = dict(
    inp="#ECEFF1", enc="#E3F2FD", back="#EDE7F6",
    style="#FFE0B2", styleE="#E65100",
    fm="#B2DFDB", fmE="#00695C",
    out="#F1F8E9", knob="#FFF176", train="#FAFAFA",
    grey="#9E9E9E",
)

fig, ax = plt.subplots(figsize=(17, 8))
ax.set_xlim(-0.3, 22.2)
ax.set_ylim(-7.4, 4.4)
ax.axis("off")
ax.set_aspect("equal")


def box(x0, y0, x1, y1, fc, ec="#555555", lw=1.0, r=0.12, z=2):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(p)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def txt(x, y, s, fs=11, color="black", w="normal", z=5, ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=w, ha=ha, va=va, zorder=z)


def arrow(p0, p1, color="#666666", lw=1.4, z=1):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        lw=lw, color=color, zorder=z,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ---- inputs ----
box(0.0, 0.5, 2.3, 1.8, C["inp"]); txt(1.15, 1.15, "Multi-view\ncameras", 11)
box(0.0, -1.8, 2.3, -0.5, C["inp"]); txt(1.15, -1.15, "Instruction · ego\nstate · sys. prompt", 9.5)

# ---- encoders ----
box(3.0, 0.55, 5.1, 1.75, C["enc"]); txt(4.05, 1.15, "Vision encoder", 10.5)
box(3.0, -1.75, 5.1, -0.55, C["enc"]); txt(4.05, -1.15, "Text tokenizer", 10.5)

# ---- backbone ----
box(5.8, -0.95, 8.1, 0.95, C["back"], lw=1.4)
txt(6.95, 0.25, "VLA backbone", 11.5, w="bold")
txt(6.95, -0.35, "(Qwen2.5-VL-3B)", 9.5, color="#444")

# ---- content tokens ----
box(8.6, -0.8, 10.5, 0.8, "#D1C4E9")
txt(9.55, 0.15, "Content\ntokens  h", 10.5, w="bold")
txt(9.55, -0.55, "(+reasoning, greyed)", 7.5, color=C["grey"])

# ---- style task-vector block ----
sx = box(10.95, -1.35, 14.35, 1.55, C["style"], ec=C["styleE"], lw=1.6)
txt(12.65, 1.25, "Style as a task vector", 11.5, w="bold", color=C["styleE"])
txt(12.65, 0.55, r"persona adapters  $\{\Delta_i\}$", 10)
txt(12.65, 0.12, "(LoRA, distilled from teachers)", 8, color="#555")
txt(12.65, -0.7, r"$c = h + \sum_i w_i\,\alpha\,\Delta_i$", 12)

# knob
ax.add_patch(Circle((12.65, -2.5), 0.62, facecolor=C["knob"], edgecolor="#555", lw=1.0, zorder=3))
txt(12.65, -2.5, r"$\alpha$/persona", 8.5)
txt(12.65, -3.4, 'the "You" knob', 8.5, color=C["styleE"], w="bold")
arrow((12.65, -1.88), (12.65, -1.37), color=C["styleE"])

# ---- flow-matching decoder ----
box(14.9, -1.75, 18.9, 1.9, C["fm"], ec=C["fmE"], lw=1.6)
txt(16.9, 1.55, "Flow-matching decoder", 11.5, w="bold", color=C["fmE"])
txt(16.9, 0.85, "noise → velocity field", 9.5)
txt(16.9, 0.45, "(DiT/AdaLN), conditioned on c", 8, color="#555")
txt(16.9, -0.15, r"control space $(a,\omega)$", 9.5)
txt(16.9, -0.55, "→ unicycle integrate", 9.5)
txt(16.9, -1.25, "N = 16 feasible candidates", 9.5, w="bold")

# crossed-out codebook
box(15.7, 2.35, 18.1, 3.15, "#FDECEA", ec="#E57373", lw=1.0)
txt(16.9, 2.75, "discrete action\ncodebook", 8.5)
ax.plot([15.7, 18.1], [2.35, 3.15], color="#E53935", lw=1.4, zorder=6)
ax.plot([15.7, 18.1], [3.15, 2.35], color="#E53935", lw=1.4, zorder=6)
txt(19.15, 2.75, "replaced", 8.5, color="#E53935", ha="left")

# ---- commit / output ----
box(19.4, -1.25, 21.9, 1.4, C["out"])
txt(20.65, 1.05, "commit one", 11, w="bold")
txt(20.65, 0.6, "candidate set → one", 8, color="#555")
# trajectory sketch
xs = [19.7, 20.1, 20.6, 21.1, 21.6]
ax.plot(xs, [-0.2, 0.05, 0.15, 0.1, -0.05], color="#BBB", lw=1.2, zorder=6)
ax.plot(xs, [-0.2, -0.35, -0.3, -0.45, -0.6], color="#BBB", lw=1.2, zorder=6)
ax.plot(xs, [-0.2, -0.1, 0.0, -0.1, -0.25], color=C["fmE"], lw=2.0, zorder=7)
txt(20.65, -0.95, "final trajectory", 9)

# ---- main arrows ----
arrow((2.3, 1.15), (3.0, 1.15))
arrow((2.3, -1.15), (3.0, -1.15))
arrow((5.1, 1.15), (5.8, 0.45))
arrow((5.1, -1.15), (5.8, -0.45))
arrow((8.1, 0.0), (8.6, 0.0))
arrow((10.5, 0.0), (10.95, 0.0))
arrow((14.35, 0.0), (14.9, 0.0), color=C["fmE"])
txt(14.62, 0.28, "c", 10, color=C["fmE"])
arrow((18.9, 0.0), (19.4, 0.0))

# ---- training strip ----
box(-0.1, -6.95, 21.9, -4.75, C["train"], ec="#CCCCCC", lw=1.0, r=0.15, z=0)
txt(0.15, -4.55, "Training", 12, w="bold", ha="left")
box(0.4, -6.65, 10.3, -5.15, "#EDE7F6", z=1)
txt(5.35, -5.7,
    "1. Base training:  fine-tune backbone + flow-matching\n"
    "decoder on human GT trajectories (vision encoder frozen).",
    9.5)
box(10.8, -6.65, 21.6, -5.15, "#FFE0B2", z=1)
txt(16.2, -5.7,
    "2. Persona distillation:  freeze the base; train one\n"
    r"low-rank adapter per teacher planner  →  task vector $\Delta_i$.",
    9.5)

plt.tight_layout()
out = "/root/workspace/CVPR_youdrive/figures/pipeline_mockup.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
