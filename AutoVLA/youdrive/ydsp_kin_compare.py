"""Kinematic YDSP (DAI/PAI) comparison: KL-anchored vs no-KL persona, across alpha.

env_interact failed in this environment (no SAI), but DAI (env weight 8%) and PAI
(env weight 21%) are kinematic-dominated and computable from trajectory dumps alone.
Env features are set to human-median (z=0) for every model -> fair, consistent.
Factors reported as percentile vs human (50 = human median, higher = more aggressive).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ST = json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
LOAD = json.load(open("youdrive/pca_style_out.json"))["loadings"]
DT = 0.5
KIN = ["v_avg", "v_std", "peak_acc", "peak_dec", "long_jerk", "lat_amax", "lat_jerk"]
ENVK = ["thw_min", "ttc_min", "dist_any", "dist_vru"]
ALL = KIN + ENVK
L = np.array([LOAD[f] for f in ALL]).T          # [3,11]
ORI = np.array([-1, 1, -1])                      # DAI=-PC1, PAI=+PC2, SAI=-PC3


def kin(vx, vy):
    vx = np.asarray(vx, float); vy = np.asarray(vy, float); sp = np.hypot(vx, vy)
    ax = np.diff(sp) / DT if len(sp) > 1 else np.array([0.])
    lj = np.diff(ax) / DT if len(ax) > 1 else np.array([0.])
    ay = np.diff(vy) / DT if len(vy) > 1 else np.array([0.])
    aj = np.diff(ay) / DT if len(ay) > 1 else np.array([0.])
    return [float(sp.mean()), float(sp.std()), float(ax.max()) if ax.size else 0,
            float(-ax.min()) if ax.size else 0, float(np.sqrt((lj ** 2).mean())) if lj.size else 0,
            float(np.abs(ay).max()) if ay.size else 0, float(np.sqrt((aj ** 2).mean())) if aj.size else 0]


def mvxvy(xy):
    p = np.array([[0., 0.]] + list(xy), float); d = np.diff(p, axis=0) / DT; return d[:, 0], d[:, 1]


# human kinematic reference
H = [kin(r["vx_ego"], r["vy_ego"]) for r in ST.values() if r.get("vx_ego") and r.get("vy_ego")]
H = np.array(H)                                   # [n,7]
HSORT = [np.sort(H[:, j]) for j in range(7)]


def kin_z(k7):
    # percentile vs human for 7 kin feats (centered), env 4 feats -> 0 (human median)
    z = np.zeros(11)
    for j in range(7):
        z[j] = np.searchsorted(HSORT[j], k7[j], side="right") / len(HSORT[j]) - 0.5
    return z


def factors(rows7):
    Z = np.array([kin_z(k) for k in rows7])
    return (Z @ L.T) * ORI                        # [n,3]


HF = factors(H)                                   # human factor distribution


def pct(val, dist):
    return float((dist < val).mean() * 100)


def model_factors(path):
    d = json.load(open(path))
    rows = [kin(*mvxvy(xy)) for t, xy in d.items() if t != "_meta"]
    F = factors(rows)
    med = np.median(F, axis=0)
    return [pct(med[k], HF[:, k]) for k in range(3)]


CF = "/mnt/pfs/zhengguantian/autovla/compare_full"
CP = "/mnt/pfs/zhengguantian/autovla/compare_persona"
CK = "/mnt/pfs/zhengguantian/autovla/compare_persona_kl20"

POINTS = [
    ("base (alpha=0)",   f"{CF}/autovla.json"),
    ("DDv2 teacher",     f"{CF}/diffusiondrivev2.json"),
    ("noKL alpha=0.5",   f"{CP}/persona_a0.5.json"),
    ("noKL alpha=1.0",   f"{CP}/persona_a1.0.json"),
    ("KL20 alpha=0.5",   f"{CK}/persona_a0.5.json"),
    ("KL20 alpha=0.75",  f"{CK}/persona_a0.75.json"),
    ("KL20 alpha=1.0",   f"{CK}/persona_a1.0.json"),
]

print(f"{'model':<18}{'DAI':>7}{'PAI':>7}{'SAI*':>7}   (percentile vs human; 50=human; higher=more aggressive)")
print("-" * 60)
res = {}
for name, path in POINTS:
    dai, pai, sai = model_factors(path)
    res[name] = (dai, pai, sai)
    print(f"{name:<18}{dai:>7.1f}{pai:>7.1f}{sai:>7.1f}")
print("\n* SAI unreliable (env features missing; SAI is 73% env-driven). DAI/PAI valid.")

# plot DAI & PAI vs alpha: noKL vs KL, with base & DDv2 reference lines
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, fi, fname in [(axes[0], 0, "DAI (dynamic aggressiveness)"), (axes[1], 1, "PAI (pace aggressiveness)")]:
    base = res["base (alpha=0)"][fi]
    ddv2 = res["DDv2 teacher"][fi]
    nokl_a = [0.0, 0.5, 1.0]
    nokl_v = [base, res["noKL alpha=0.5"][fi], res["noKL alpha=1.0"][fi]]
    kl_a = [0.0, 0.5, 0.75, 1.0]
    kl_v = [base, res["KL20 alpha=0.5"][fi], res["KL20 alpha=0.75"][fi], res["KL20 alpha=1.0"][fi]]
    ax.plot(nokl_a, nokl_v, "o-", color="#e6194B", lw=2.2, ms=8, label="no-KL")
    ax.plot(kl_a, kl_v, "s-", color="#4363d8", lw=2.2, ms=8, label="KL=20")
    ax.axhline(ddv2, color="#3cb44b", ls="--", lw=1.8, label=f"DDv2 target ({ddv2:.0f})")
    ax.axhline(50, color="#999", ls=":", lw=1.2, label="human (50)")
    ax.set_title(fname, fontsize=12)
    ax.set_xlabel("alpha (persona strength)"); ax.set_ylabel("percentile vs human")
    ax.set_xticks([0, 0.5, 0.75, 1.0]); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
out = "/mnt/pfs/zhengguantian/autovla/persona/ydsp_kl_vs_nokl.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
json.dump(res, open("youdrive/ydsp_kin_compare_out.json", "w"), indent=2)
