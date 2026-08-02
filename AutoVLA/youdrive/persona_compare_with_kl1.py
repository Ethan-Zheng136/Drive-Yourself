"""Persona comparison extended with KL=1 (beta=1) anchor.

Adds the beta=1 series alongside no-KL / KL=5 / KL=20 across the 5 env social
metrics + DAI/PAI/SAI YDSP factors. Reuses the exact factor pipeline from
persona_compare_full.py (PCA loadings + human-percentile projection).

Outputs:
  - printed social-metric table (all alphas) for kl1 vs kl5 vs kl20 vs no-KL
  - printed DAI/PAI/SAI factor table per alpha
  - figure: /mnt/pfs/zhengguantian/autovla/persona/persona_compare_with_kl1.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_fp = "/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
try:
    fm.fontManager.addfont(_fp); plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

ST = json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ENV1 = json.load(open("youdrive/env_interact_out.json"))      # base/DDv2/human/no-KL persona
ENV2 = json.load(open("youdrive/env_interact_kl20.json"))     # KL=20 persona
ENV3 = json.load(open("youdrive/env_interact_kl5.json"))      # KL=5 persona
ENV4 = json.load(open("youdrive/env_interact_kl1.json"))      # KL=1 persona
LOAD = json.load(open("youdrive/pca_style_out.json"))["loadings"]
DT = 0.5
CF = "/mnt/pfs/zhengguantian/autovla/compare_full"
CP = "/mnt/pfs/zhengguantian/autovla/compare_persona"
CK = "/mnt/pfs/zhengguantian/autovla/compare_persona_kl20"
CK5 = "/mnt/pfs/zhengguantian/autovla/compare_persona_kl5"
CK1 = "/mnt/pfs/zhengguantian/autovla/compare_persona_kl1"

ENV_METRICS = ["thw_min", "ttc_min", "lead_gap_min", "dist_any_min", "dist_vru_min"]


def med_env(env_dict, key, metric):
    d = env_dict.get(key, {})
    v = [r[metric] for r in d.values() if isinstance(r, dict) and r.get(metric) is not None]
    return float(np.median(v)) if v else float("nan")


KIN = ["v_avg", "v_std", "peak_acc", "peak_dec", "long_jerk", "lat_amax", "lat_jerk"]
ENVK = ["thw_min", "ttc_min", "dist_any", "dist_vru"]
ENVK_SRC = ["thw_min", "ttc_min", "dist_any_min", "dist_vru_min"]
ALLF = KIN + ENVK
L = np.array([LOAD[f] for f in ALLF]).T
ORI = np.array([-1, 1, -1])


def kin(vx, vy):
    vx = np.asarray(vx, float); vy = np.asarray(vy, float); sp = np.hypot(vx, vy)
    ax = np.diff(sp) / DT if len(sp) > 1 else np.array([0.])
    lj = np.diff(ax) / DT if len(ax) > 1 else np.array([0.])
    ay = np.diff(vy) / DT if len(vy) > 1 else np.array([0.])
    aj = np.diff(ay) / DT if len(ay) > 1 else np.array([0.])
    return {"v_avg": float(sp.mean()), "v_std": float(sp.std()), "peak_acc": float(ax.max()) if ax.size else 0,
            "peak_dec": float(-ax.min()) if ax.size else 0, "long_jerk": float(np.sqrt((lj**2).mean())) if lj.size else 0,
            "lat_amax": float(np.abs(ay).max()) if ay.size else 0, "lat_jerk": float(np.sqrt((aj**2).mean())) if aj.size else 0}


def mvxvy(xy):
    p = np.array([[0., 0.]] + list(xy), float); d = np.diff(p, axis=0) / DT; return d[:, 0], d[:, 1]


Hf = []
for t, r in ST.items():
    if not r.get("vx_ego"):
        continue
    f = kin(r["vx_ego"], r["vy_ego"])
    e = ENV1["human"].get(t, {})
    for lk, sk in zip(ENVK, ENVK_SRC):
        f[lk] = e.get(sk)
    Hf.append(f)
HSORT = {}
for k in ALLF:
    v = np.array([d[k] for d in Hf if d.get(k) is not None], float); HSORT[k] = np.sort(v)


def pctile(val, arr):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return np.searchsorted(arr, val, side="right") / len(arr) - 0.5


def factor_rows(dump_path, env_dict, env_key):
    d = json.load(open(dump_path)); ed = env_dict.get(env_key, {}); rows = []
    for t, xy in d.items():
        if t == "_meta":
            continue
        f = kin(*mvxvy(xy)); e = ed.get(t, {})
        for lk, sk in zip(ENVK, ENVK_SRC):
            f[lk] = e.get(sk)
        z = np.array([pctile(f[k], HSORT[k]) for k in ALLF])
        rows.append((z @ L.T) * ORI)
    return np.array(rows)


HZ = np.array([[pctile(d[k], HSORT[k]) for k in ALLF] for d in Hf])
HFAC = (HZ @ L.T) * ORI


def pct_vs_human(med, k):
    return float((HFAC[:, k] < med[k]).mean() * 100)


def factors_pct(dump_path, env_dict, env_key):
    R = factor_rows(dump_path, env_dict, env_key)
    med = np.median(R, axis=0)
    return [pct_vs_human(med, k) for k in range(3)]


# ---- series definitions (alpha, env_dict, env_key, dump_path) ----
NOKL = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
        (0.5, ENV1, "persona_a0.5", f"{CP}/persona_a0.5.json"),
        (1.0, ENV1, "persona_a1.0", f"{CP}/persona_a1.0.json")]
KL1 = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
       (0.5, ENV4, "kl1_a0.5", f"{CK1}/persona_a0.5.json"),
       (0.75, ENV4, "kl1_a0.75", f"{CK1}/persona_a0.75.json"),
       (1.0, ENV4, "kl1_a1.0", f"{CK1}/persona_a1.0.json")]
KL5 = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
       (0.5, ENV3, "kl5_a0.5", f"{CK5}/persona_a0.5.json"),
       (0.75, ENV3, "kl5_a0.75", f"{CK5}/persona_a0.75.json"),
       (1.0, ENV3, "kl5_a1.0", f"{CK5}/persona_a1.0.json")]
KL20 = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
        (0.5, ENV2, "kl20_a0.5", f"{CK}/persona_a0.5.json"),
        (0.75, ENV2, "kl20_a0.75", f"{CK}/persona_a0.75.json"),
        (1.0, ENV2, "kl20_a1.0", f"{CK}/persona_a1.0.json")]
DDV2_env = "diffusiondrivev2"
DDV2_dump = f"{CF}/diffusiondrivev2.json"


def series_env(points, metric):
    return [med_env(ed, ek, metric) for (_, ed, ek, _) in points]


def series_fac(points, fi):
    return [factors_pct(dp, ed, ek)[fi] for (_, ed, ek, dp) in points]


ddv2_env = {m: med_env(ENV1, DDV2_env, m) for m in ENV_METRICS}
ddv2_fac = factors_pct(DDV2_dump, ENV1, DDV2_env)
human_env = {m: med_env(ENV1, "human", m) for m in ENV_METRICS}

# ---- print social-metric table per alpha ----
print("=== env social metrics (median over valid scenes), by KL anchor & alpha ===")
ALPHAS = [0.5, 0.75, 1.0]
SERIES = {"noKL": NOKL, "KL1": KL1, "KL5": KL5, "KL20": KL20}
for m in ENV_METRICS:
    print(f"\n[{m}]   human={human_env[m]:.2f}  base={med_env(ENV1,'autovla',m):.2f}  DDv2={ddv2_env[m]:.2f}")
    hdr = "  ".join(f"a{a}" for a in ALPHAS)
    print(f"  {'anchor':<6} " + "  ".join(f"{'a'+str(a):>6}" for a in ALPHAS))
    for name, pts in SERIES.items():
        amap = {p[0]: (p[1], p[2]) for p in pts}
        vals = []
        for a in ALPHAS:
            if a in amap:
                ed, ek = amap[a]; vals.append(f"{med_env(ed, ek, m):>6.2f}")
            else:
                vals.append(f"{'-':>6}")
        print(f"  {name:<6} " + "  ".join(vals))

print("\n\n=== YDSP factors (percentile vs human; 50=human-like, higher=more aggressive) ===")
fn = ["DAI", "PAI", "SAI"]
for name, pts in SERIES.items():
    print(f"\n--- {name} ---")
    print(f"  {'alpha':<6}{'DAI':>8}{'PAI':>8}{'SAI':>8}")
    for (a, ed, ek, dp) in pts:
        if a == 0.0:
            continue
        fac = factors_pct(dp, ed, ek)
        print(f"  {a:<6}{fac[0]:>8.1f}{fac[1]:>8.1f}{fac[2]:>8.1f}")
base_fac = factors_pct(f"{CF}/autovla.json", ENV1, "autovla")
print(f"\n  base   {base_fac[0]:>8.1f}{base_fac[1]:>8.1f}{base_fac[2]:>8.1f}")
print(f"  DDv2   {ddv2_fac[0]:>8.1f}{ddv2_fac[1]:>8.1f}{ddv2_fac[2]:>8.1f}")
print("  human      50.0    50.0    50.0")

# ---- plot 2x4 ----
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
panels = [(m, "env") for m in ENV_METRICS] + [(i, "fac") for i in range(3)]
titles = {"thw_min": "THW 车头时距(s)", "ttc_min": "min-TTC(s)", "lead_gap_min": "前车间距(m)",
          "dist_any_min": "最近任意距离(m)", "dist_vru_min": "最近VRU距离(m)",
          0: "DAI 动态激进(pct)", 1: "PAI 速度激进(pct)", 2: "SAI 社交激进(pct)"}
nokl_a = [p[0] for p in NOKL]; kl1_a = [p[0] for p in KL1]; kl5_a = [p[0] for p in KL5]; kl20_a = [p[0] for p in KL20]
for ax, (key, kind) in zip(axes.flat, panels):
    if kind == "env":
        nk = series_env(NOKL, key); k1 = series_env(KL1, key); k5 = series_env(KL5, key); k20 = series_env(KL20, key)
        dd = ddv2_env[key]; hu = human_env[key]
    else:
        nk = series_fac(NOKL, key); k1 = series_fac(KL1, key); k5 = series_fac(KL5, key); k20 = series_fac(KL20, key)
        dd = ddv2_fac[key]; hu = 50.0
    ax.plot(nokl_a, nk, "o-", color="#e6194B", lw=2.2, ms=7, label="no-KL")
    ax.plot(kl1_a, k1, "D-", color="#911eb4", lw=2.2, ms=7, label="KL=1")
    ax.plot(kl5_a, k5, "^-", color="#f58231", lw=2.2, ms=7, label="KL=5")
    ax.plot(kl20_a, k20, "s-", color="#4363d8", lw=2.2, ms=7, label="KL=20")
    ax.axhline(dd, color="#3cb44b", ls="--", lw=1.8, label="DDv2 target")
    ax.axhline(hu, color="#999", ls=":", lw=1.2, label="human")
    ax.set_title(titles[key], fontsize=12); ax.set_xlabel("α"); ax.grid(alpha=0.3)
    ax.set_xticks([0, 0.5, 0.75, 1.0])
axes.flat[0].legend(fontsize=9, loc="best")
fig.suptitle("Persona 风格 dose-response:无KL (红) vs KL=1 (紫) vs KL=5 (橙) vs KL=20 (蓝) vs DDv2 (绿虚) vs 人类 (灰点)",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "/mnt/pfs/zhengguantian/autovla/persona/persona_compare_with_kl1.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print("\nsaved", out)
