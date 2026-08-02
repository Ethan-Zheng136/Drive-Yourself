"""Behavioral / temporal driving-style metrics across the 8 planner "drivers".

Question: kinematic metrics (jerk, accel) discriminate the 8 planners very well
(CV 65-112%), but static social-spacing metrics (thw/ttc/distances) do NOT
(CV 2-6%) -- all planners are safety-trained and keep similar spacing.
Hypothesis: real social/assertiveness STYLE lives in behavioral-temporal
DECISIONS (how readily the car goes when the path is clear / from a stop / into
a gap), not in static spacing. This script builds + tests those metrics.

It reuses the env_interact.py machinery for scene/agent access and lead
detection:
  - youdrive/env_interact.py:40  agents_in_t0(scene) -> per future-frame
        (agent_xy, agent_vel, is_vru) in the t=0 ego frame.
  - youdrive/env_interact.py:63  ego_series(xy)      -> (positions, speed)
        with the t=0 ego position prepended at the origin.
  - youdrive/env_interact.py:80-88  lead = agent ahead (x>ego_x) within a
        +-CORRIDOR(1.5 m) lateral band; lead_gap None  <=>  no lead in horizon.
We re-use the SAME lead rule per-step to define "path is clear / lead-free".

Behavioral metrics (per token -> per-driver median -> CV% across drivers):
  HESITATION  (free-flow tokens only; "free-flow" = GT/human path has NO lead
               anywhere in the 4 s horizon, i.e. env_interact lead_gap is None):
     hes_decel_mean  mean longitudinal braking magnitude  mean(max(0,-a_long))   [m/s^2]
     hes_decel_frac  fraction of free-flow steps with a_long < -0.3 m/s^2        [0-1]
     hes_speed_cv    within-token speed coeff-of-variation  std(sp)/mean(sp)     [-]
       higher = slows / wobbles more with nothing in front = more hesitant.
  START DELAY (tokens where the REAL ego is near-stationary at t=0,
               speed_0 < 1 m/s, AND free-flow -> shared token set for all drivers):
     start_t2        time (s) for the planned traj to reach 2 m/s (cap = horizon)
     start_dist2s    distance (m) travelled in first 2 s
       higher start_t2 / lower start_dist2s = slower to get moving = more hesitant.
  GAP ACCEPTANCE (heuristic, lane-change scenes; reported with reliability caveat):
     accepted_gap    min distance (m) to nearest "alongside" agent during the
                     lateral maneuver.  smaller = more assertive.

Kinematic baselines (peak_dec, long_jerk, v_avg) are recomputed here with the
identical per-driver-median -> CV pipeline to validate the 65-112% reference
number and give an apples-to-apples comparison.

Outputs -> /mnt/pfs (repo rule: never write large artifacts locally).
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
sys.path.insert(0, AV)
sys.path.insert(0, f"{AV}/navsim")

# reuse env_interact's pure helpers (agents_in_t0, ego_series) without triggering
# its module-level dump loading: point ONLY_MODELS at a single tiny dump.
CMP = "/mnt/pfs/zhengguantian/autovla/compare_full"
os.environ.setdefault("ONLY_MODELS", json.dumps({"autovla": f"{CMP}/autovla.json"}))
import importlib
EI = importlib.import_module("youdrive.env_interact")
agents_in_t0 = EI.agents_in_t0
ego_series   = EI.ego_series
CORRIDOR     = EI.CORRIDOR        # 1.5 m lateral half-width (same lead rule)

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

DATA = "/root/workspace/closed_loop/data/navsim"
ST   = json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
DT   = 0.5
KMAX = 8                          # cap horizon to 8 pts (4.0 s) so autovla(10) is fair
CAP  = int(os.environ.get("CAP", "99999"))
OUT  = os.environ.get("OUT", "/mnt/pfs/zhengguantian/autovla/persona/behavioral_metrics_out.json")

# the 8 "drivers" (planner dumps). gtrs/hydra_mdp use the *_r resampled dumps.
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

# thresholds
V_STATIONARY = 1.0    # m/s  -> "near-stationary" qualifying scene for start-delay
V_GO         = 2.0    # m/s  -> "moving" threshold for start_t2
DECEL_THR    = 0.3    # m/s^2 -> a step counts as "braking" if a_long < -DECEL_THR
LC_LAT       = 1.8    # m    -> |net lateral| above this (human) = lane-change scene
ALONGSIDE_DX = 3.0    # m    -> |long. offset| below this = agent is "beside" ego


def speed_accel(xy):
    """speed series (len K) and longitudinal accel series (len K-1) for a dump traj."""
    xy = np.asarray(xy, float)[:KMAX]
    _, sp = ego_series(xy)                       # env_interact.py:63 (prepends origin)
    a = np.diff(sp) / DT if len(sp) > 1 else np.array([0.0])
    return sp, a


def lead_free_token(ego_xy, agents):
    """True iff NO lead vehicle anywhere in horizon, using env_interact's lead rule.
    Lead = agent ahead (x>ego_x) within +-CORRIDOR lateral band  (env_interact.py:80-88)."""
    pos, _ = ego_series(np.asarray(ego_xy, float)[:KMAX])
    K = min(len(pos), len(agents))
    for k in range(K):
        exy = pos[k]; axy = agents[k][0]
        if len(axy) == 0:
            continue
        ahead = (axy[:, 0] > exy[0]) & (np.abs(axy[:, 1] - exy[1]) < CORRIDOR)
        if ahead.any():
            return False
    return True


def accepted_gap(ego_xy, agents):
    """Heuristic gap acceptance for a lane-change traj: min distance to an
    'alongside' agent (|long offset|<ALONGSIDE_DX) during the max-|lateral-velocity|
    step. Returns (gap_or_None, had_alongside_agent)."""
    pos, _ = ego_series(np.asarray(ego_xy, float)[:KMAX])
    if len(pos) < 2:
        return None, False
    lat_v = np.abs(np.diff(pos[:, 1])) / DT      # |lateral speed| per step
    kc = int(np.argmax(lat_v)) + 1               # step where the cross happens
    kc = min(kc, len(pos) - 1, len(agents) - 1)
    exy = pos[kc]; axy = agents[kc][0]
    if len(axy) == 0:
        return None, False
    alongside = np.abs(axy[:, 0] - exy[0]) < ALONGSIDE_DX
    if not alongside.any():
        return None, False
    d = np.hypot(axy[alongside, 0] - exy[0], axy[alongside, 1] - exy[1])
    return float(d.min()), True


def kin_feats(xy):
    """peak_dec (m/s^2), long_jerk RMS (m/s^3), v_avg (m/s) -- kinematic baseline."""
    sp, a = speed_accel(xy)
    lj = np.diff(a) / DT if len(a) > 1 else np.array([0.0])
    return {
        "peak_dec":  float(-a.min()) if a.size else 0.0,
        "long_jerk": float(np.sqrt((lj ** 2).mean())) if lj.size else 0.0,
        "v_avg":     float(sp.mean()) if sp.size else 0.0,
    }


def main():
    toks = [t for t in MODELS["autovla"] if t != "_meta"]
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                     has_route=True, max_scenes=None, log_names=None, tokens=toks)
    loader = SceneLoader(data_path=Path(f"{DATA}/navsim_logs/test"),
                         sensor_blobs_path=Path(f"{DATA}/sensor_blobs/test"),
                         scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    common = [t for t in loader.tokens if t in ST][:CAP]
    print("scenes to process:", len(common), flush=True)

    # accumulators: per driver -> metric -> list of per-token values
    hes  = {d: {"hes_decel_mean": [], "hes_decel_frac": [], "hes_speed_cv": []} for d in DRIVERS}
    strt = {d: {"start_t2": [], "start_dist2s": []} for d in DRIVERS}
    gap  = {d: {"accepted_gap": []} for d in DRIVERS}
    kin  = {d: {"peak_dec": [], "long_jerk": [], "v_avg": []} for d in DRIVERS}
    n_freeflow = 0; n_stationary_ff = 0; n_lanechange = 0; n_lc_with_agent = 0

    nh = 4
    for i, tok in enumerate(common):
        if i % 500 == 0:
            print(f"  {i}/{len(common)}", flush=True)
        try:
            sc = loader.get_scene_from_token(tok)
            ag = agents_in_t0(sc)                                  # env_interact.py:40
            hum_xy = np.asarray(sc.get_future_trajectory().poses)[:, :2]
            v0 = float(np.hypot(*sc.frames[nh - 1].ego_status.ego_velocity))
        except Exception as e:
            print("skip", tok, repr(e)[:90]); continue

        is_freeflow   = lead_free_token(hum_xy, ag)
        is_stationary = v0 < V_STATIONARY
        # lane-change scene flagged on the REAL (human) trajectory's net lateral shift
        is_lanechange = (abs(hum_xy[KMAX - 1, 1] if len(hum_xy) >= KMAX else hum_xy[-1, 1]) > LC_LAT
                         and (hum_xy[-1, 0] > 2.0))
        if is_freeflow:   n_freeflow += 1
        if is_freeflow and is_stationary: n_stationary_ff += 1
        if is_lanechange: n_lanechange += 1
        lc_agent_seen = False

        for d in DRIVERS:
            dump = MODELS[d]
            if tok not in dump:
                continue
            xy = dump[tok]
            sp, a = speed_accel(xy)

            # always-on kinematic baseline
            kf = kin_feats(xy)
            for kk, vv in kf.items():
                kin[d][kk].append(vv)

            # HESITATION (free-flow only)
            if is_freeflow and a.size:
                decel = np.maximum(0.0, -a)
                hes[d]["hes_decel_mean"].append(float(decel.mean()))
                hes[d]["hes_decel_frac"].append(float((a < -DECEL_THR).mean()))
                if sp.mean() > 1e-3:
                    hes[d]["hes_speed_cv"].append(float(sp.std() / sp.mean()))

            # START DELAY (near-stationary + free-flow; shared token set)
            if is_stationary and is_freeflow and sp.size:
                hit = np.where(sp >= V_GO)[0]
                t2 = float((hit[0] + 1) * DT) if hit.size else float(KMAX * DT)
                strt[d]["start_t2"].append(t2)
                n2 = min(int(round(2.0 / DT)), len(xy))      # first 2 s of positions
                dist2 = float(np.hypot(*np.asarray(xy[n2 - 1], float))) if n2 >= 1 else 0.0
                strt[d]["start_dist2s"].append(dist2)

            # GAP ACCEPTANCE (lane-change scenes; heuristic)
            if is_lanechange:
                g, had = accepted_gap(xy, ag)
                if had:
                    gap[d]["accepted_gap"].append(g); lc_agent_seen = True
        if is_lanechange and lc_agent_seen:
            n_lc_with_agent += 1

    # ---- per-driver medians + CV% across drivers --------------------------------
    def med(lst):
        return float(np.median(lst)) if len(lst) else float("nan")

    def cv(vals):
        v = np.asarray([x for x in vals if not math.isnan(x)], float)
        if len(v) < 2 or abs(v.mean()) < 1e-12:
            return float("nan")
        return float(100.0 * v.std() / abs(v.mean()))     # population std (np default)

    report = {
        "_def": {
            "free_flow": "GT/human path has NO lead (env_interact lead rule) over 4s horizon",
            "hes_decel_mean": "mean(max(0,-a_long)) over free-flow steps [m/s^2]; higher=hesitant",
            "hes_decel_frac": "fraction free-flow steps with a_long<-0.3 m/s^2; higher=hesitant",
            "hes_speed_cv": "within-token std(speed)/mean(speed) on free-flow tokens; higher=hesitant",
            "start_t2": "time(s) to reach 2 m/s from near-stationary free-flow start (cap 4s); higher=hesitant",
            "start_dist2s": "distance(m) travelled in first 2s from stationary start; LOWER=hesitant",
            "accepted_gap": "min dist(m) to alongside agent during lane-change (heuristic); smaller=assertive",
            "kin_*": "kinematic baseline recomputed with same pipeline for CV validation",
            "CV": "100*std/mean across the 8 per-driver medians (population std)",
        },
        "counts": {
            "n_scenes": len(common), "n_freeflow": n_freeflow,
            "n_stationary_freeflow": n_stationary_ff,
            "n_lanechange_scenes": n_lanechange, "n_lanechange_with_alongside_agent": n_lc_with_agent,
        },
        "per_driver": {}, "cv": {},
    }

    blocks = [("HESITATION", hes), ("START_DELAY", strt), ("GAP_ACCEPTANCE", gap), ("KINEMATIC_baseline", kin)]
    for _, acc in blocks:
        for d in DRIVERS:
            report["per_driver"].setdefault(d, {})
            for m, lst in acc[d].items():
                report["per_driver"][d][m] = med(lst)

    metric_order = []
    for _, acc in blocks:
        for m in acc[DRIVERS[0]]:
            metric_order.append(m)
    for m in metric_order:
        report["cv"][m] = cv([report["per_driver"][d][m] for d in DRIVERS])

    json.dump(report, open(OUT, "w"), indent=2)

    # ---- pretty print -----------------------------------------------------------
    print("\n=== scene counts ===")
    for k, v in report["counts"].items():
        print(f"  {k:<34}{v}")

    print("\n=== per-driver median  +  CV% across drivers ===")
    hdr = f"{'metric':<16}" + "".join(f"{d[:9]:>10}" for d in DRIVERS) + f"{'CV%':>8}"
    print(hdr); print("-" * len(hdr))
    for m in metric_order:
        row = f"{m:<16}"
        for d in DRIVERS:
            row += f"{report['per_driver'][d][m]:>10.3f}"
        row += f"{report['cv'][m]:>8.1f}"
        print(row)

    print("\n=== behavioral metrics ranked by CV% (discriminability) ===")
    beh = [m for blk, acc in blocks[:3] for m in acc[DRIVERS[0]]]
    for m in sorted(beh, key=lambda x: (-report['cv'][x] if not math.isnan(report['cv'][x]) else 1e9)):
        print(f"  {m:<16}{report['cv'][m]:>7.1f}%")
    print("\n  (kinematic baseline for reference:)")
    for m in ["peak_dec", "long_jerk", "v_avg"]:
        print(f"  {m:<16}{report['cv'][m]:>7.1f}%")
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
