"""Safety (PDMS) vs Style (ttc_min: lower = more aggressive/DDv2-like) at alpha=1.0.
Shows every approach as a point; the target corner is DDv2 (high PDMS + low ttc)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
try:
    fp = "/mnt/pfs/zhengguantian/autovla/NotoSansCJKsc-Regular.otf"
    fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

ENV1 = json.load(open("youdrive/env_interact_out.json"))
ENVK = json.load(open("youdrive/env_interact_kl20.json"))
ENVG = json.load(open("youdrive/env_interact_grpo.json"))


def ttc(env, key):
    d = env.get(key, {})
    v = [r["ttc_min"] for r in d.values() if isinstance(r, dict) and r.get("ttc_min") is not None]
    return float(np.median(v)) if v else float("nan")


# (label, PDMS@a1.0, ttc_min@a1.0, color, marker)
pts = [
    ("base (no style)",      0.8946, ttc(ENV1, "autovla"),          "#888888", "o"),
    ("DDv2 teacher (target)",0.9108, ttc(ENV1, "diffusiondrivev2"), "#3cb44b", "*"),
    ("no-KL SFT",            0.7373, ttc(ENV1, "persona_a1.0"),     "#e6194B", "o"),
    ("KL=20",                0.8818, ttc(ENVK, "kl20_a1.0"),        "#4363d8", "s"),
    ("GRPO (CE+PDMS)",       0.7159, ttc(ENVG, "grpo_a1.0"),        "#f032e6", "D"),
]
human_ttc = ttc(ENV1, "human")

fig, ax = plt.subplots(figsize=(9, 7))
for lbl, pdms, t, c, m in pts:
    ax.scatter(t, pdms, c=c, s=260 if m == "*" else 160, marker=m, edgecolors="k", linewidths=0.8, zorder=3, label=lbl)
    ax.annotate(f"{lbl}\nPDMS={pdms:.3f}, ttc={t:.2f}", (t, pdms),
                textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.axhline(0.87, color="#999", ls=":", lw=1.2)
ax.text(ax.get_xlim()[0], 0.872, " safe zone (PDMS>=0.87)", color="#666", fontsize=9, va="bottom")
ax.axvline(human_ttc, color="#bbb", ls=":", lw=1)
ax.set_xlabel("style axis: min-TTC (s)  <-- more aggressive / DDv2-like", fontsize=12)
ax.set_ylabel("safety axis: PDMS (alpha=1.0)", fontsize=12)
ax.set_title("Safety vs Style @ alpha=1.0\nGoal = DDv2 corner (low ttc + high PDMS); GRPO landed = no-KL SFT (style yes, unsafe)",
             fontsize=12, fontweight="bold")
ax.invert_xaxis()  # left = more aggressive style
ax.grid(alpha=0.3)
plt.tight_layout()
out = "/mnt/pfs/zhengguantian/autovla/persona/safety_vs_style.png"
plt.savefig(out, dpi=130, bbox_inches="tight")

print(f"{'config':<22}{'PDMS':>8}{'ttc_min':>9}  (style: lower ttc = more DDv2-like)")
for lbl, pdms, t, c, m in pts:
    print(f"{lbl:<22}{pdms:>8.3f}{t:>9.2f}")
print(f"{'human':<22}{'-':>8}{human_ttc:>9.2f}")
print("\nsaved", out)
