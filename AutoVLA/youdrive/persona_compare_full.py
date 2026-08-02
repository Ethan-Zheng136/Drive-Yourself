"""Full persona comparison: KL=20 vs no-KL, across alpha, vs base & DDv2 target.
Panels: 5 env social metrics (raw medians) + 3 YDSP factors (DAI/PAI/SAI, percentile vs human).
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
LOAD = json.load(open("youdrive/pca_style_out.json"))["loadings"]
DT = 0.5
CF = "/mnt/pfs/zhengguantian/autovla/compare_full"
CP = "/mnt/pfs/zhengguantian/autovla/compare_persona"
CK = "/mnt/pfs/zhengguantian/autovla/compare_persona_kl20"

# ---- raw env social metrics (median over valid scenes) ----
ENV_METRICS = ["thw_min", "ttc_min", "lead_gap_min", "dist_any_min", "dist_vru_min"]


def med_env(env_dict, key, metric):
    d = env_dict.get(key, {})
    v = [r[metric] for r in d.values() if isinstance(r, dict) and r.get(metric) is not None]
    return float(np.median(v)) if v else float("nan")


# ---- YDSP factors (with real env) ----
KIN = ["v_avg", "v_std", "peak_acc", "peak_dec", "long_jerk", "lat_amax", "lat_jerk"]
ENVK = ["thw_min", "ttc_min", "dist_any", "dist_vru"]      # loading feature names
ENVK_SRC = ["thw_min", "ttc_min", "dist_any_min", "dist_vru_min"]  # env_interact output names
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


# human reference (kin from ST + env from ENV1["human"])
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


HF = factor_rows  # placeholder
# human factor distribution
HZ = np.array([[pctile(d[k], HSORT[k]) for k in ALLF] for d in Hf])
HFAC = (HZ @ L.T) * ORI


def pct_vs_human(med, k):
    return float((HFAC[:, k] < med[k]).mean() * 100)


def factors_pct(dump_path, env_dict, env_key):
    R = factor_rows(dump_path, env_dict, env_key)
    med = np.median(R, axis=0)
    return [pct_vs_human(med, k) for k in range(3)]


# ---- assemble points ----
# (label, alpha, env_dict, env_key, dump_path)
NOKL = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
        (0.5, ENV1, "persona_a0.5", f"{CP}/persona_a0.5.json"),
        (1.0, ENV1, "persona_a1.0", f"{CP}/persona_a1.0.json")]
KL = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
      (0.5, ENV2, "kl20_a0.5", f"{CK}/persona_a0.5.json"),
      (0.75, ENV2, "kl20_a0.75", f"{CK}/persona_a0.75.json"),
      (1.0, ENV2, "kl20_a1.0", f"{CK}/persona_a1.0.json")]
CK5 = "/mnt/pfs/zhengguantian/autovla/compare_persona_kl5"
KL5 = [(0.0, ENV1, "autovla", f"{CF}/autovla.json"),
       (0.5, ENV3, "kl5_a0.5", f"{CK5}/persona_a0.5.json"),
       (0.75, ENV3, "kl5_a0.75", f"{CK5}/persona_a0.75.json"),
       (1.0, ENV3, "kl5_a1.0", f"{CK5}/persona_a1.0.json")]
DDV2_env = "diffusiondrivev2"
DDV2_dump = f"{CF}/diffusiondrivev2.json"

# ---- compute ----
def series_env(points, metric):
    return [med_env(ed, ek, metric) for (_, ed, ek, _) in points]

def series_fac(points, fi):
    return [factors_pct(dp, ed, ek)[fi] for (_, ed, ek, dp) in points]

ddv2_env = {m: med_env(ENV1, DDV2_env, m) for m in ENV_METRICS}
ddv2_fac = factors_pct(DDV2_dump, ENV1, DDV2_env)
human_env = {m: med_env(ENV1, "human", m) for m in ENV_METRICS}

# print table
print("=== env social metrics (median, alpha=1.0) ===")
print(f"{'metric':<14}{'base':>8}{'noKL':>8}{'KL5':>8}{'KL20':>8}{'DDv2':>8}{'human':>8}")
for m in ENV_METRICS:
    b = med_env(ENV1, "autovla", m); nk = med_env(ENV1, "persona_a1.0", m)
    k5 = med_env(ENV3, "kl5_a1.0", m); kl = med_env(ENV2, "kl20_a1.0", m)
    print(f"{m:<14}{b:>8.2f}{nk:>8.2f}{k5:>8.2f}{kl:>8.2f}{ddv2_env[m]:>8.2f}{human_env[m]:>8.2f}")
print("\n=== YDSP factors (percentile vs human, 50=human) ===")
fn = ["DAI", "PAI", "SAI"]
print(f"{'factor':<6}{'base':>8}{'noKL_a1':>9}{'KL_a1':>8}{'DDv2':>8}")
base_fac = factors_pct(f"{CF}/autovla.json", ENV1, "autovla")
nokl_fac = factors_pct(f"{CP}/persona_a1.0.json", ENV1, "persona_a1.0")
kl_fac = factors_pct(f"{CK}/persona_a1.0.json", ENV2, "kl20_a1.0")
for i, nm in enumerate(fn):
    print(f"{nm:<6}{base_fac[i]:>8.1f}{nokl_fac[i]:>9.1f}{kl_fac[i]:>8.1f}{ddv2_fac[i]:>8.1f}")

# ---- plot 2x4 ----
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
panels = [(m, "env") for m in ENV_METRICS] + [(i, "fac") for i in range(3)]
titles = {"thw_min": "THW 车头时距(s)", "ttc_min": "min-TTC(s)", "lead_gap_min": "前车间距(m)",
          "dist_any_min": "最近任意距离(m)", "dist_vru_min": "最近VRU距离(m)",
          0: "DAI 动态激进(pct)", 1: "PAI 速度激进(pct)", 2: "SAI 社交激进(pct)"}
nokl_a = [p[0] for p in NOKL]; kl_a = [p[0] for p in KL]; kl5_a = [p[0] for p in KL5]
for ax, (key, kind) in zip(axes.flat, panels):
    if kind == "env":
        nk = series_env(NOKL, key); kl = series_env(KL, key); k5 = series_env(KL5, key)
        dd = ddv2_env[key]; hu = human_env[key]
    else:
        nk = series_fac(NOKL, key); kl = series_fac(KL, key); k5 = series_fac(KL5, key)
        dd = ddv2_fac[key]; hu = 50.0
    ax.plot(nokl_a, nk, "o-", color="#e6194B", lw=2.2, ms=7, label="no-KL")
    ax.plot(kl5_a, k5, "^-", color="#f58231", lw=2.2, ms=7, label="KL=5")
    ax.plot(kl_a, kl, "s-", color="#4363d8", lw=2.2, ms=7, label="KL=20")
    ax.axhline(dd, color="#3cb44b", ls="--", lw=1.8, label="DDv2 target")
    ax.axhline(hu, color="#999", ls=":", lw=1.2, label="human")
    ax.set_title(titles[key], fontsize=12); ax.set_xlabel("α"); ax.grid(alpha=0.3)
    ax.set_xticks([0, 0.5, 0.75, 1.0])
axes.flat[0].legend(fontsize=9, loc="best")
fig.suptitle("Persona 风格 dose-response:无KL (红) vs KL=5 (橙) vs KL=20 (蓝) vs DDv2老师 (绿虚) vs 人类 (灰点)", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "/mnt/pfs/zhengguantian/autovla/persona/persona_compare_full_kl5_kl20.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print("\nsaved", out)
