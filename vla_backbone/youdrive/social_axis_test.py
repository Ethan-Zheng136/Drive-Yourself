"""DEFINITIVE test: does a SECOND "social style" axis exist for the 8 planner
"drivers" that is BOTH (a) discriminative across drivers AND (b) decorrelated
from the kinematic/dynamics axis (per-driver longitudinal-jerk RMS)?

This EXTENDS youdrive/behavioral_metrics.py. Prior coarse runs were
inconclusive because they used the wrong scenes / granularity:
  - aggregate social spacing over ALL tokens was flat (CV 2-6%) because most
    tokens are open-road (no interaction) -> differences diluted;
  - coarse start_delay (2.0 s cap, 101 tokens) was flat (CV 8.5%).
Here we measure two refined candidates PROPERLY:

  (1) SOCIAL-ON-DENSE-SUBSET -- restrict to interaction-dense tokens (a real
      lead exists, OR >=k agents within R m of ego at t0) and only THERE take
      per-driver medians of thw_min / ttc_min / lead_gap_min / dist_any_min.
  (2) SCENARIO-HESITATION (fine) -- on standing-start tokens (v0<1 m/s) and on
      low-speed clear-path tokens, measure FINE (sub-step, interpolated) time
      to reach 0.5 / 1.0 / 2.0 m/s (no 2 s cap) + mean accel over first 1.5 s.

For every refined metric we report: per-driver median, CV% across the 8
drivers, and Pearson correlation of the metric's per-driver vector vs the
dynamics axis (per-driver long_jerk). A metric is a REAL second social axis
ONLY if CV >= ~40% (discriminative, comparable to acc_avg) AND |corr| < ~0.5
(orthogonal to dynamics).

Reused machinery (cite file:line):
  - youdrive/env_interact.py:40  agents_in_t0(scene)  (agents in t=0 ego frame)
  - youdrive/env_interact.py:63  ego_series(xy)       (positions + speed)
  - youdrive/env_interact.py:69  interact(ego,agents) (thw/ttc/lead_gap/dist)
  - youdrive/env_interact.py:25/80-88  CORRIDOR + lead rule
  - youdrive/behavioral_metrics.py:92/135  speed/accel + kin_feats conventions

Outputs -> /mnt/pfs (repo rule: never write large artifacts locally).
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
sys.path.insert(0, AV)
sys.path.insert(0, f"{AV}/navsim")

CMP = "/mnt/pfs/zhengguantian/autovla/compare_full"
# load env_interact pure helpers WITHOUT triggering its module-level multi-dump load
os.environ.setdefault("ONLY_MODELS", json.dumps({"autovla": f"{CMP}/autovla.json"}))
import importlib
EI = importlib.import_module("youdrive.env_interact")
agents_in_t0 = EI.agents_in_t0       # env_interact.py:40
ego_series   = EI.ego_series         # env_interact.py:63
interact     = EI.interact           # env_interact.py:69
CORRIDOR     = EI.CORRIDOR           # env_interact.py:25 (1.5 m)

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

DATA = "/root/workspace/closed_loop/data/navsim"
ST   = json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
DT   = 0.5
KMAX = 8                                   # cap horizon to 8 pts (4.0 s) -> fair vs autovla(10)
CAP  = int(os.environ.get("CAP", "99999"))
OUT  = os.environ.get("OUT", "/mnt/pfs/zhengguantian/autovla/persona/social_axis_out.json")

DUMP = {
    "autovla":          f"{CMP}/autovla.json",
    "diffusiondrive":   f"{CMP}/diffusiondrive.json",
    "diffusiondrivev2": f"{CMP}/diffusiondrivev2.json",
    "transfuser":       f"{CMP}/transfuser.json",
    "gtrs":             f"{CMP}/gtrs_r.json",
    "hydra_mdp":        f"{CMP}/hydra_mdp_r.json",
    "wote":             f"{CMP}/wote.json",
    "goalflow":         f"{CMP}/goalflow.json",
}
DRIVERS = list(DUMP)
MODELS = {k: json.load(open(v)) for k, v in DUMP.items()}

# ---- thresholds -------------------------------------------------------------
V_STAND   = 1.0                    # m/s  standing-start: real ego speed0 < this
GO_LEVELS = [0.5, 1.0, 2.0]        # m/s  fine time-to-move crossing levels
ACC_WIN   = 1.5                    # s    window for mean start accel
# density-subset definitions: (label, predicate(scene_flags)->bool)
DENSE_DEFS = [
    ("lead_GT",      lambda f: f["lead_GT"]),               # human GT has a lead in horizon
    ("ag1_R20",      lambda f: f["n20"] >= 1),              # >=1 agent within 20 m at t0
    ("agg2_R20",     lambda f: f["n20"] >= 2),              # >=2 agents within 20 m at t0
    ("agg3_R15",     lambda f: f["n15"] >= 3),              # >=3 agents within 15 m at t0
]
SOC_METRICS = ["thw_min", "ttc_min", "lead_gap_min", "dist_any_min"]


# ---- per-trajectory primitives (behavioral_metrics.py:92/135 conventions) ---
def speed_accel(xy):
    xy = np.asarray(xy, float)[:KMAX]
    _, sp = ego_series(xy)                          # env_interact.py:63
    a = np.diff(sp) / DT if len(sp) > 1 else np.array([0.0])
    return sp, a


def kin_feats(xy):
    """dynamics-axis primitives: long_jerk RMS, lat_jerk RMS, acc_avg, v_avg."""
    xy = np.asarray(xy, float)[:KMAX]
    pos, sp = ego_series(xy)
    a = np.diff(sp) / DT if len(sp) > 1 else np.array([0.0])
    lj = np.diff(a) / DT if len(a) > 1 else np.array([0.0])
    # lateral jerk from the cross-track (y) channel of cumulative positions
    if len(pos) > 3:
        ay = np.diff(pos[:, 1], n=2) / (DT * DT)
        ljat = np.diff(ay) / DT
    else:
        ljat = np.array([0.0])
    return {
        "long_jerk": float(np.sqrt((lj ** 2).mean())) if lj.size else 0.0,
        "lat_jerk":  float(np.sqrt((ljat ** 2).mean())) if ljat.size else 0.0,
        "acc_avg":   float(np.abs(a).mean()) if a.size else 0.0,
        "v_avg":     float(sp.mean()) if sp.size else 0.0,
    }


def speed_curve(xy, v0):
    """speed-vs-time samples with the real t=0 speed prepended.
    sp[k] = avg speed over interval k -> assign to midpoint (k+0.5)*DT (most
    accurate for a monotone standing-start ramp); prepend (t=0, v0)."""
    _, sp = ego_series(np.asarray(xy, float)[:KMAX])
    t = (np.arange(len(sp)) + 0.5) * DT
    tt = np.concatenate([[0.0], t])
    ss = np.concatenate([[v0], sp])
    return tt, ss


def time_to_speed(tt, ss, V):
    """FINE (sub-step linear interpolation) time at which speed first reaches V.
    Returns nan if V never reached within the horizon."""
    if ss[0] >= V:
        return 0.0
    for i in range(1, len(ss)):
        if ss[i] >= V:
            d = ss[i] - ss[i - 1]
            frac = (V - ss[i - 1]) / d if abs(d) > 1e-9 else 0.0
            return float(tt[i - 1] + frac * (tt[i] - tt[i - 1]))
    return float("nan")


# ---- stats helpers ----------------------------------------------------------
def med(lst):
    v = [x for x in lst if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def cv(vals):
    v = np.asarray([x for x in vals if not math.isnan(x)], float)
    if len(v) < 2 or abs(v.mean()) < 1e-12:
        return float("nan")
    return float(100.0 * v.std() / abs(v.mean()))      # population std, across the 8 drivers


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    toks = [t for t in MODELS["autovla"] if t != "_meta"]
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                     has_route=True, max_scenes=None, log_names=None, tokens=toks)
    loader = SceneLoader(data_path=Path(f"{DATA}/navsim_logs/test"),
                         sensor_blobs_path=Path(f"{DATA}/sensor_blobs/test"),
                         scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    common = [t for t in loader.tokens if t in ST][:CAP]
    print("scenes to process:", len(common), flush=True)

    # per-token scene flags (driver-independent: defined from GT + agents)
    flags = {}                                          # tok -> {lead_GT,n10,n15,n20,standing,v0}
    # per (driver,tok) measured quantities
    soc = {d: {} for d in DRIVERS}                      # tok -> interact dict
    kin = {d: {"long_jerk": [], "lat_jerk": [], "acc_avg": [], "v_avg": []} for d in DRIVERS}
    hes = {d: {} for d in DRIVERS}                      # tok -> {t05,t10,t20,acc15}

    nh = 4
    for i, tok in enumerate(common):
        if i % 500 == 0:
            print(f"  {i}/{len(common)}", flush=True)
        try:
            sc = loader.get_scene_from_token(tok)
            ag = agents_in_t0(sc)                                   # env_interact.py:40
            hum_xy = np.asarray(sc.get_future_trajectory().poses)[:, :2]
            v0 = float(np.hypot(*sc.frames[nh - 1].ego_status.ego_velocity))
        except Exception as e:
            print("skip", tok, repr(e)[:90]); continue

        # agent density at t0: use the earliest future frame (closest to t0),
        # count agents within R m of the ego (origin in the t=0 frame).
        n10 = n15 = n20 = 0
        if len(ag):
            axy0 = ag[0][0]
            if len(axy0):
                dr = np.hypot(axy0[:, 0], axy0[:, 1])
                n10 = int((dr < 10).sum()); n15 = int((dr < 15).sum()); n20 = int((dr < 20).sum())
        lead_GT = interact(hum_xy, ag).get("lead_gap_min") is not None    # env_interact.py:69
        flags[tok] = {"lead_GT": lead_GT, "n10": n10, "n15": n15, "n20": n20,
                      "standing": v0 < V_STAND, "v0": v0}

        for d in DRIVERS:
            dump = MODELS[d]
            if tok not in dump:
                continue
            xy = dump[tok]
            # dynamics axis (always-on, over ALL tokens)
            for kk, vv in kin_feats(xy).items():
                kin[d][kk].append(vv)
            # social interaction metrics (subset-filtered later)
            soc[d][tok] = interact(np.asarray(xy), ag)
            # fine hesitation (computed for all; aggregated on standing subset)
            tt, ss = speed_curve(xy, v0)
            v_at = float(np.interp(ACC_WIN, tt, ss))
            hes[d][tok] = {
                "t05": time_to_speed(tt, ss, GO_LEVELS[0]),
                "t10": time_to_speed(tt, ss, GO_LEVELS[1]),
                "t20": time_to_speed(tt, ss, GO_LEVELS[2]),
                "acc15": (v_at - v0) / ACC_WIN,
            }

    # ===== dynamics axis vector (per-driver median long_jerk over ALL tokens) =
    dyn_vec = {d: float(np.median(kin[d]["long_jerk"])) for d in DRIVERS}
    dyn_list = [dyn_vec[d] for d in DRIVERS]
    kin_meds = {d: {k: float(np.median(v)) for k, v in kin[d].items()} for d in DRIVERS}

    report = {
        "_def": {
            "dynamics_axis": "per-driver median longitudinal-jerk RMS over ALL tokens",
            "qualify": "social axis iff CV>=40% (discriminative) AND |corr_vs_long_jerk|<0.5 (orthogonal)",
            "dense_defs": {
                "lead_GT": "human GT path has a lead in 4s horizon (env_interact lead rule)",
                "agg1_R20/agg2_R20/agg3_R15": ">=k agents within R m of ego at t0 (earliest future frame)",
            },
            "soc_metrics": "thw_min/ttc_min/lead_gap_min (lower=closer/assertive), dist_any_min",
            "hesitation": "FINE interpolated time(s) to reach 0.5/1.0/2.0 m/s (no cap); acc15=(v@1.5s - v0)/1.5",
        },
        "counts": {}, "dynamics_axis_long_jerk": dyn_vec,
        "kin_baseline_per_driver": kin_meds, "kin_baseline_cv": {},
        "dense_subset": {}, "hesitation": {},
    }
    for k in ["long_jerk", "lat_jerk", "acc_avg", "v_avg"]:
        report["kin_baseline_cv"][k] = cv([kin_meds[d][k] for d in DRIVERS])

    # ===== MEASUREMENT 1: SOCIAL-ON-DENSE-SUBSET =============================
    n_total = len(flags)
    for label, pred in DENSE_DEFS:
        sub = [t for t, f in flags.items() if pred(f)]
        report["counts"][f"dense[{label}]"] = len(sub)
        block = {"n_tokens": len(sub), "per_driver": {}, "cv": {}, "corr_vs_long_jerk": {}}
        for d in DRIVERS:
            block["per_driver"][d] = {}
            for m in SOC_METRICS:
                block["per_driver"][d][m] = med([soc[d].get(t, {}).get(m) for t in sub])
        for m in SOC_METRICS:
            vec = [block["per_driver"][d][m] for d in DRIVERS]
            block["cv"][m] = cv(vec)
            block["corr_vs_long_jerk"][m] = pearson(vec, dyn_list)
        report["dense_subset"][label] = block

    # ===== MEASUREMENT 2: SCENARIO-HESITATION (fine) =========================
    stand = [t for t, f in flags.items() if f["standing"]]
    # low-speed clear-path: slow (v0<3) AND no lead in GT horizon (clear path)
    lowclear = [t for t, f in flags.items() if f["v0"] < 3.0 and not f["lead_GT"]]
    report["counts"]["standing(v0<1)"] = len(stand)
    report["counts"]["lowspeed_clearpath(v0<3,no-lead)"] = len(lowclear)
    HES_METRICS = ["t05", "t10", "t20", "acc15"]
    for label, sub in [("standing_start", stand), ("lowspeed_clearpath", lowclear)]:
        block = {"n_tokens": len(sub), "per_driver": {}, "cv": {}, "corr_vs_long_jerk": {},
                 "reach_frac": {}}
        for d in DRIVERS:
            block["per_driver"][d] = {}
            for m in HES_METRICS:
                block["per_driver"][d][m] = med([hes[d].get(t, {}).get(m) for t in sub])
            # fraction of standing tokens that actually reach 2.0 m/s within horizon
            present = [t for t in sub if t in hes[d]]
            t20s = [hes[d][t].get("t20") for t in present]
            fin = [x for x in t20s if x is not None and not math.isnan(x)]
            block["reach_frac"][d] = round(len(fin) / len(present), 3) if present else float("nan")
        for m in HES_METRICS:
            vec = [block["per_driver"][d][m] for d in DRIVERS]
            block["cv"][m] = cv(vec)
            block["corr_vs_long_jerk"][m] = pearson(vec, dyn_list)
        report["hesitation"][label] = block

    report["counts"]["n_total"] = n_total
    json.dump(report, open(OUT, "w"), indent=2)

    # ===== pretty print ======================================================
    def qualtag(cvv, corr):
        if math.isnan(cvv) or math.isnan(corr):
            return "  n/a"
        disc = cvv >= 40.0; orth = abs(corr) < 0.5
        return " QUALIFIES" if (disc and orth) else ("  no(disc)" if not disc else "  no(corr)")

    print("\n================ DYNAMICS AXIS (reference) ================")
    print(f"{'driver':<18}{'long_jerk':>10}{'lat_jerk':>10}{'acc_avg':>10}{'v_avg':>10}")
    for d in DRIVERS:
        k = kin_meds[d]
        print(f"{d:<18}{k['long_jerk']:>10.3f}{k['lat_jerk']:>10.3f}{k['acc_avg']:>10.3f}{k['v_avg']:>10.3f}")
    print(f"{'CV%':<18}" + "".join(f"{report['kin_baseline_cv'][k]:>10.1f}"
                                    for k in ['long_jerk', 'lat_jerk', 'acc_avg', 'v_avg']))

    print("\n================ MEASUREMENT 1: SOCIAL-ON-DENSE-SUBSET ================")
    for label, _ in DENSE_DEFS:
        b = report["dense_subset"][label]
        print(f"\n--- dense subset [{label}]  n_tokens={b['n_tokens']} ---")
        hdr = f"{'metric':<14}" + "".join(f"{d[:8]:>9}" for d in DRIVERS) + f"{'CV%':>8}{'corr':>8}  verdict"
        print(hdr); print("-" * len(hdr))
        for m in SOC_METRICS:
            row = f"{m:<14}"
            for d in DRIVERS:
                val = b["per_driver"][d][m]
                row += f"{val:>9.3f}" if not math.isnan(val) else f"{'--':>9}"
            row += f"{b['cv'][m]:>8.1f}{b['corr_vs_long_jerk'][m]:>8.2f}"
            row += qualtag(b['cv'][m], b['corr_vs_long_jerk'][m])
            print(row)

    print("\n================ MEASUREMENT 2: SCENARIO-HESITATION (fine) ================")
    for label in ["standing_start", "lowspeed_clearpath"]:
        b = report["hesitation"][label]
        print(f"\n--- {label}  n_tokens={b['n_tokens']} ---")
        hdr = f"{'metric':<14}" + "".join(f"{d[:8]:>9}" for d in DRIVERS) + f"{'CV%':>8}{'corr':>8}  verdict"
        print(hdr); print("-" * len(hdr))
        for m in ["t05", "t10", "t20", "acc15"]:
            row = f"{m:<14}"
            for d in DRIVERS:
                val = b["per_driver"][d][m]
                row += f"{val:>9.3f}" if not math.isnan(val) else f"{'--':>9}"
            row += f"{b['cv'][m]:>8.1f}{b['corr_vs_long_jerk'][m]:>8.2f}"
            row += qualtag(b['cv'][m], b['corr_vs_long_jerk'][m])
            print(row)
        print("reach2.0 frac:", {d: b["reach_frac"][d] for d in DRIVERS})

    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
