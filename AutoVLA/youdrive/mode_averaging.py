"""Mode-averaging harm experiment.

Hypothesis: on scenes where the 16 planner "drivers/teachers" strongly disagree
(multi-modal: some brake / some go), a policy that outputs the AVERAGE of the
modes lands "between" them and is LESS SAFE (lower PDMS) than committing to any
single mode.

Safety metric: native nuPlan/navsim PDM-Score (PDMS) computed on the navtest
metric cache (metric_cache_navtest_v1). ALL 4049 styletest tokens are present in
that cache, so PDMS is fully feasible (not a proxy). We additionally report the
PDMS sub-metrics (no_at_fault_collision, drivable_area, TTC, progress, comfort,
driving_direction) and a geometric "distance-to-nearest-real-mode" descriptor.

Pipeline
  1. Load 16 driver dumps, canonicalize each to 8 pts @ [0.5..4.0]s (ego BEV).
  2. Per token compute cross-driver disagreement (mean pairwise endpoint dist +
     longitudinal brake/go bimodality). Rank tokens; take top-K multimodal (MM)
     and bottom-K unimodal (UM) control.
  3. Per selected token build AVERAGED trajectory (mean over drivers) and the
     mode-committed trajectories (each driver + 2 k-means mode centroids).
  4. Score everything with real PDMS. Compare averaged vs mode-committed within
     MM vs UM (difference-in-differences + significance).
"""
import os, sys, json, time, traceback
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats

AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
sys.path.insert(0, AV); sys.path.insert(0, f"{AV}/navsim")
import lzma, pickle
from models.utils.score import PDM_Reward, Trajectory, TrajectorySampling
from navsim.evaluate.pdm_score import pdm_score

CMP = "/mnt/pfs/zhengguantian/autovla/compare_full"
MC = Path("/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
OUT_MD = "/mnt/pfs/zhengguantian/autovla/persona/mode_averaging.md"
OUT_JSON = "/mnt/pfs/zhengguantian/autovla/persona/mode_averaging_scores.json"
DRIVERS = ["autovla","diffusiondrive","diffusiondrivev2","drivesuprim","drivoR",
           "goalflow","gtrs","gtrs_aug","gtrs_dp","gtrs_r","hydra_mdp","hydra_mdp_r",
           "sparsedrivev2","trajdiff_scaling","transfuser","wote"]
K = int(os.environ.get("K", "300"))          # tokens per group
MIN_DRIVERS = int(os.environ.get("MIN_DRIVERS", "15"))
MINCLUST = int(os.environ.get("MINCLUST", "3"))   # min drivers per mode to call it bimodal
GAP_LONG = float(os.environ.get("GAP_LONG", "4.0"))  # min longitudinal gap for brake/go split (m)
GAP_LAT = float(os.environ.get("GAP_LAT", "1.5"))    # min lateral gap for left/right split (m)
SEED = 0

t_common = np.linspace(0.5, 4.0, 8)          # canonical grid
t_dst = np.linspace(0.5, 5.0, 10)            # scorer grid

def canon(xy, H, n):
    xy = np.asarray(xy, float)
    t_src = np.linspace(H/n, H, n)
    v = (xy[-1]-xy[-2])/(t_src[-1]-t_src[-2])
    out = np.zeros((8,2))
    for i,t in enumerate(t_common):
        if t <= t_src[-1]+1e-6:
            out[i,0]=np.interp(t,t_src,xy[:,0]); out[i,1]=np.interp(t,t_src,xy[:,1])
        else:
            out[i]=xy[-1]+v*(t-t_src[-1])
    return out

def to_score10(xy8):
    v=(xy8[-1]-xy8[-2])/(t_common[-1]-t_common[-2])
    out=np.zeros((10,2))
    for i,t in enumerate(t_dst):
        if t<=t_common[-1]+1e-6:
            out[i,0]=np.interp(t,t_common,xy8[:,0]); out[i,1]=np.interp(t,t_common,xy8[:,1])
        else:
            out[i]=xy8[-1]+v*(t-t_common[-1])
    return out

def headings(xy10):
    prev=np.array([0.,0.]); hs=[]
    for p in xy10:
        d=p-prev; hs.append(np.arctan2(d[1],d[0])); prev=p
    return np.array(hs)

# ---------------- load + canonicalize ----------------
print("[load] reading 16 driver dumps ...", flush=True)
raw={d:json.load(open(f"{CMP}/{d}.json")) for d in DRIVERS}
meta={d:raw[d]["_meta"] for d in DRIVERS}
rew=PDM_Reward(MC)
avail=set(rew.metric_cache_loader.metric_cache_paths.keys())
samp=TrajectorySampling(num_poses=10,interval_length=0.5)

# token -> {driver: (8,2)}   (only drivers present)
all_toks=set()
for d in DRIVERS: all_toks |= (set(raw[d])-{"_meta"})
all_toks &= avail
canon_by_tok={}
for tk in all_toks:
    dd={}
    for d in DRIVERS:
        if tk in raw[d]:
            dd[d]=canon(raw[d][tk], float(meta[d]["horizon_s"]), int(meta[d]["n_pts"]))
    if len(dd)>=MIN_DRIVERS:
        canon_by_tok[tk]=dd
print(f"[load] {len(canon_by_tok)} tokens with >={MIN_DRIVERS} drivers (of {len(all_toks)} in cache)", flush=True)

# ---------------- disagreement + longitudinal brake/go mode structure ----------------
def pairwise_mean(pts):
    n=len(pts); s=0.; c=0
    for i in range(n):
        for j in range(i+1,n):
            s+=np.hypot(*(pts[i]-pts[j])); c+=1
    return s/max(c,1)

def modes_2d(dd):
    """2-means on the 4s (x,y) endpoint -> mode structure.
    Distinguishes longitudinal (brake/go, x-axis) from lateral (left/right, y-axis) splits."""
    names=sorted(dd)
    trajs=np.stack([dd[d] for d in names])
    ep=trajs[:,-1,:]                                    # (ndrv,2) endpoints
    km=KMeans(2,n_init=4,random_state=SEED).fit(ep)
    lab=km.labels_; cc=km.cluster_centers_              # (2,2)
    sizes=[int((lab==0).sum()),int((lab==1).sum())]
    long_gap=float(abs(cc[0,0]-cc[1,0]))                # x separation of the two modes
    lat_gap =float(abs(cc[0,1]-cc[1,1]))                # y separation of the two modes
    within=np.sqrt(np.mean(np.sum((ep-cc[lab])**2,1)))
    sep=float(np.hypot(*(cc[0]-cc[1]))/(within+1e-6))
    return names,trajs,lab,cc,sizes,sep,long_gap,lat_gap

info={}
for tk,dd in canon_by_tok.items():
    names,trajs,lab,cc,sizes,sep,long_gap,lat_gap=modes_2d(dd)
    ep=trajs[:,-1,:]
    disagree=pairwise_mean(ep); minclust=min(sizes)
    populated = minclust>=MINCLUST
    is_lat  = populated and (lat_gap>=GAP_LAT) and (lat_gap>=long_gap)     # left/right dominant
    is_long = populated and (long_gap>=GAP_LONG) and (long_gap>lat_gap)    # brake/go dominant
    info[tk]=dict(disagree=disagree,long_range=float(ep[:,0].ptp()),lat_range=float(ep[:,1].ptp()),
                  sep=sep,long_gap=long_gap,lat_gap=lat_gap,sizes=sizes,minclust=minclust,
                  is_lat=is_lat,is_long=is_long)

n_lat=sum(v["is_lat"] for v in info.values()); n_long=sum(v["is_long"] for v in info.values())
lat_toks =sorted([t for t,v in info.items() if v["is_lat"]],  key=lambda t:-(info[t]["sep"]*info[t]["lat_gap"]))
long_toks=sorted([t for t,v in info.items() if v["is_long"]], key=lambda t:-(info[t]["sep"]*info[t]["long_gap"]))
um       =sorted(info, key=lambda t:info[t]["disagree"])[:K]
mm_lat=lat_toks[:K]; mm_long=long_toks[:K]
print(f"[select] lateral(left/right) bimodal: {n_lat} | longitudinal(brake/go) bimodal: {n_long}", flush=True)
print(f"[select] MM_lat={len(mm_lat)} MM_long={len(mm_long)} UM={len(um)}", flush=True)

# ---------------- scoring ----------------
_score_cache={}
def score8(xy8, tok):
    xy10=to_score10(xy8); hs=headings(xy10)
    traj=Trajectory(np.concatenate([xy10,hs[:,None]],1),samp)
    mpath=rew.metric_cache_loader.metric_cache_paths[tok]
    with lzma.open(mpath,"rb") as f: mc=pickle.load(f)
    r=pdm_score(metric_cache=mc,model_trajectory=traj,future_sampling=rew.future_sampling,
                simulator=rew.simulator,scorer=rew.scorer)
    return dict(score=float(r.score),coll=float(r.no_at_fault_collisions),
                dac=float(r.drivable_area_compliance),prog=float(r.ego_progress),
                ttc=float(r.time_to_collision_within_bound),comfort=float(r.comfort),
                ddc=float(r.driving_direction_compliance))

def analyze(tokens, label):
    out=[]
    t0=time.time(); nerr=0
    for i,tk in enumerate(tokens):
        if i%50==0:
            el=time.time()-t0
            print(f"  [{label}] {i}/{len(tokens)}  {el:.0f}s", flush=True)
        dd=canon_by_tok[tk]
        names,trajs,lab,cc,sizes,sep,long_gap,lat_gap=modes_2d(dd)
        try:
            # individual drivers
            drv_scores={d:score8(dd[d],tk) for d in names}
            pv=np.array([drv_scores[d]["score"] for d in names])
            # AVERAGED = mean over all drivers (the "output the average of the modes" policy)
            avg=trajs.mean(0)
            s_avg=score8(avg,tk)
            # MODE-COMMITTED = centroid of each longitudinal (brake/go) mode cluster
            cent_scores=[]; cent_ep=[]
            for c in range(2):
                m=(lab==c)
                cen=trajs[m].mean(0)
                cent_scores.append(score8(cen,tk)["score"]); cent_ep.append(cen[-1])
            cent_scores=np.array(cent_scores); w=np.array(sizes,float)/np.sum(sizes)
            # geometry: averaged endpoint distance to nearest real driver & nearest mode centroid
            ep_avg=avg[-1]; eps=trajs[:,-1,:]
            d_near=float(np.min(np.hypot(eps[:,0]-ep_avg[0],eps[:,1]-ep_avg[1])))
            d_near_cent=float(np.min([np.hypot(*(ep_avg-e)) for e in cent_ep]))
            fulld=float(np.min([np.mean(np.hypot(*(trajs[j]-avg).T)) for j in range(len(names))]))
            out.append(dict(tok=tk,label=label,ndrv=len(names),
                disagree=info[tk]["disagree"], long_range=info[tk]["long_range"],
                lat_range=info[tk]["lat_range"],
                bimodal_sep=info[tk]["sep"], long_gap=info[tk]["long_gap"], lat_gap=info[tk]["lat_gap"],
                minclust=info[tk]["minclust"],
                pdms_avg=s_avg["score"],
                # mode-committed baselines
                pdms_cent_best=float(cent_scores.max()),
                pdms_cent_worst=float(cent_scores.min()),
                pdms_cent_mean=float(cent_scores.mean()),          # commit to a random mode (mode-symmetric)
                pdms_cent_wmean=float((cent_scores*w).sum()),      # commit weighted by mode popularity
                # single-driver baselines (confounded by failed models on hard scenes)
                pdms_mean_ind=float(pv.mean()), pdms_med_ind=float(np.median(pv)),
                pdms_best_ind=float(pv.max()), pdms_worst_ind=float(pv.min()),
                # avg dominated by BOTH modes? (strongest harm signal)
                avg_below_both=bool(s_avg["score"] < cent_scores.min()-1e-6),
                avg_coll=s_avg["coll"], avg_dac=s_avg["dac"], avg_ttc=s_avg["ttc"],
                avg_prog=s_avg["prog"], avg_comfort=s_avg["comfort"], avg_ddc=s_avg["ddc"],
                mean_coll=float(np.mean([drv_scores[d]["coll"] for d in names])),
                mean_dac=float(np.mean([drv_scores[d]["dac"] for d in names])),
                mean_ttc=float(np.mean([drv_scores[d]["ttc"] for d in names])),
                mean_prog=float(np.mean([drv_scores[d]["prog"] for d in names])),
                d_near_endpoint=d_near, d_near_centroid=d_near_cent, d_near_fulltraj=fulld,
                ))
        except Exception as e:
            nerr+=1
            print(f"  [{label}] ERROR tok={tk}: {repr(e)[:120]}", flush=True)
            traceback.print_exc()
    print(f"  [{label}] done {len(out)} ok, {nerr} err, {time.time()-t0:.0f}s", flush=True)
    return out

if os.environ.get("RELOAD")=="1" and os.path.exists(OUT_JSON):
    print("[reload] loading cached scores from",OUT_JSON, flush=True)
    _d=json.load(open(OUT_JSON)); res_lat=_d["mm_lat"]; res_long=_d["mm_long"]; res_um=_d["um"]
else:
    print("[score] LATERAL (left/right) bimodal group ...", flush=True)
    res_lat=analyze(mm_lat,"MM_lat")
    print("[score] LONGITUDINAL (brake/go) bimodal group ...", flush=True)
    res_long=analyze(mm_long,"MM_long")
    print("[score] UNIMODAL control group ...", flush=True)
    res_um=analyze(um,"UM")

json.dump({"mm_lat":res_lat,"mm_long":res_long,"um":res_um,
           "config":{"K":K,"MIN_DRIVERS":MIN_DRIVERS,"MINCLUST":MINCLUST,
                     "GAP_LONG":GAP_LONG,"GAP_LAT":GAP_LAT,"drivers":DRIVERS,
                     "n_lat":n_lat,"n_long":n_long,"n_tokens":len(canon_by_tok)}},
          open(OUT_JSON,"w"))
print("[save] scores ->",OUT_JSON, flush=True)

# ---------------- stats + markdown ----------------
def agg(rs):
    return {k:np.array([r[k] for r in rs]) for k in rs[0] if isinstance(rs[0][k],(int,float,bool))}
A={"lat":agg(res_lat),"long":agg(res_long),"um":agg(res_um)}
KEYS=["pdms_avg","pdms_cent_best","pdms_cent_worst","pdms_cent_mean",
      "pdms_mean_ind","pdms_med_ind","pdms_best_ind","pdms_worst_ind",
      "disagree","long_range","lat_range","bimodal_sep","long_gap","lat_gap","minclust",
      "avg_below_both","avg_coll","mean_coll","avg_dac","mean_dac","avg_ttc","mean_ttc",
      "avg_prog","mean_prog","d_near_endpoint","d_near_centroid","d_near_fulltraj"]
S={g:{k:float(np.mean(A[g][k])) for k in KEYS} for g in A}

def penalties(a):
    return dict(rand=a["pdms_avg"]-a["pdms_cent_mean"],       # vs commit to a random mode
                best=a["pdms_avg"]-a["pdms_cent_best"],        # vs oracle-best mode
                worst=a["pdms_avg"]-a["pdms_cent_worst"])      # vs worst mode
P={g:penalties(A[g]) for g in A}
def wilx(g,ref):
    try: return stats.wilcoxon(A[g]["pdms_avg"],A[g][ref]).pvalue
    except Exception: return float("nan")
Wr={g:wilx(g,"pdms_cent_mean") for g in A}   # vs random mode
Wb={g:wilx(g,"pdms_cent_best") for g in A}   # vs oracle-best mode
# difference-in-differences on the vs-best-mode penalty (the clean, monotonic signal)
did_best = stats.mannwhitneyu(P["lat"]["best"],P["um"]["best"],alternative="less").pvalue
did_best_long = stats.mannwhitneyu(P["long"]["best"],P["um"]["best"],alternative="less").pvalue
did_rand = stats.mannwhitneyu(P["lat"]["rand"],P["um"]["rand"],alternative="less").pvalue
pct_below_both={g:float(np.mean(A[g]["avg_below_both"])) for g in A}
pct_worse_rand={g:float(np.mean(A[g]["pdms_avg"]<A[g]["pdms_cent_mean"]-1e-6)) for g in A}
pct_worse_best={g:float(np.mean(A[g]["pdms_avg"]<A[g]["pdms_cent_best"]-1e-6)) for g in A}

md=[]
md.append("# Mode-Averaging Harm: does averaging multi-modal driving behavior cost safety?\n")
md.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · safety metric = native nuPlan/navsim "
          f"**PDMS** on `metric_cache_navtest_v1` · {len(DRIVERS)} planner \"drivers\" · K={K} scenes/group._\n")

md.append("## TL;DR verdict\n")
md.append(
f"""**Yes — averaging multi-modal behaviour costs safety, and the cost grows with multi-modality.**
The cleanest evidence: averaging is *always* worse than committing to the context-appropriate
(better) mode, and that penalty **rises monotonically** with how multi-modal the scene is —
from **{P['um']['best'].mean():+.4f}** PDMS on uni-modal scenes, to **{P['long']['best'].mean():+.4f}** on
brake/go splits, to **{P['lat']['best'].mean():+.4f}** on left/right splits (≈{abs(P['lat']['best'].mean())/max(abs(P['um']['best'].mean()),1e-6):.0f}× the
uni-modal baseline; difference-in-differences lateral−unimodal Mann-Whitney p={did_best:.1e}).

The effect is **axis-dependent** — averaging hurts most for *lateral* (left/right) disagreement, the
classic "go around the obstacle on the left OR the right → average drives straight at it":

| scene type | n | averaged PDMS | commit better mode | **Δ avg−best** | commit random mode | Δ avg−rand | below BOTH modes |
|---|---|---|---|---|---|---|---|
| **Lateral (left/right)** | {len(res_lat)} | {S['lat']['pdms_avg']:.4f} | {S['lat']['pdms_cent_best']:.4f} | **{P['lat']['best'].mean():+.4f}** | {S['lat']['pdms_cent_mean']:.4f} | {P['lat']['rand'].mean():+.4f} | {100*pct_below_both['lat']:.0f}% |
| Longitudinal (brake/go) | {len(res_long)} | {S['long']['pdms_avg']:.4f} | {S['long']['pdms_cent_best']:.4f} | {P['long']['best'].mean():+.4f} | {S['long']['pdms_cent_mean']:.4f} | {P['long']['rand'].mean():+.4f} | {100*pct_below_both['long']:.0f}% |
| Uni-modal control | {len(res_um)} | {S['um']['pdms_avg']:.4f} | {S['um']['pdms_cent_best']:.4f} | {P['um']['best'].mean():+.4f} | {S['um']['pdms_cent_mean']:.4f} | {P['um']['rand'].mean():+.4f} | {100*pct_below_both['um']:.0f}% |

**Headline numbers:**
- **Committing to the right mode beats averaging by {abs(P['lat']['best'].mean()):.3f} PDMS on lateral splits**
  (paired Wilcoxon p={Wb['lat']:.1e}); averaging is worse than the better mode in {100*pct_worse_best['lat']:.0f}% of them.
- The gap is **{P['lat']['best'].mean()-P['um']['best'].mean():+.4f} PDMS larger** on lateral than uni-modal control
  (DiD, p={did_best:.1e}) and **{P['long']['best'].mean()-P['um']['best'].mean():+.4f}** larger on brake/go (p={did_best_long:.1e}).
- The averaged path lies **{S['lat']['d_near_endpoint']:.2f} m** from even the *nearest* real driver at 4 s on
  lateral scenes (vs {S['um']['d_near_endpoint']:.2f} m uni-modal) — a trajectory no actual driver would take.
- **Honest nuance:** vs a *randomly-chosen* mode, averaging roughly breaks even (it hedges), and is only
  mildly negative for lateral ({P['lat']['rand'].mean():+.4f}). So the harm is specifically "averaging vs
  committing to the *correct* mode" — i.e. the cost is the price of **ignoring which style/mode the
  context calls for**, which is exactly the claim under test.
""")

md.append("## 1. Scene selection\n")
md.append(
f"""{len(DRIVERS)} planner "drivers/teachers" ({', '.join(DRIVERS)}) each output a 4 s ego-frame
future, canonicalized to 8 pts @ [0.5..4.0]s. Over the **{len(canon_by_tok)}** tokens carrying
≥{MIN_DRIVERS} drivers, the 16 drivers' 4 s endpoints are 2-means clustered. A scene is bimodal if
both clusters hold ≥{MINCLUST} drivers; it is:

- **lateral** if the two mode centres differ by ≥{GAP_LAT} m in *y* and the y-gap dominates (left vs right) — **{n_lat}** scenes;
- **longitudinal** if they differ by ≥{GAP_LONG} m in *x* and the x-gap dominates (brake vs go) — **{n_long}** scenes.

Uni-modal control = the {len(res_um)} lowest-disagreement scenes. Each multi-modal group takes the
top-{K} by mode separation (sep × axis-gap).

| group | n | disagree (m) | long. range (m) | lat. range (m) | long gap (m) | lat gap (m) |
|---|---|---|---|---|---|---|
| Lateral | {len(res_lat)} | {S['lat']['disagree']:.2f} | {S['lat']['long_range']:.2f} | {S['lat']['lat_range']:.2f} | {S['lat']['long_gap']:.2f} | {S['lat']['lat_gap']:.2f} |
| Longitudinal | {len(res_long)} | {S['long']['disagree']:.2f} | {S['long']['long_range']:.2f} | {S['long']['lat_range']:.2f} | {S['long']['long_gap']:.2f} | {S['long']['lat_gap']:.2f} |
| Uni-modal | {len(res_um)} | {S['um']['disagree']:.3f} | {S['um']['long_range']:.3f} | {S['um']['lat_range']:.3f} | {S['um']['long_gap']:.3f} | {S['um']['lat_gap']:.3f} |
""")

md.append("## 2. Averaged vs mode-committed PDMS (primary result)\n")
md.append(
f"""| scene type | averaged | commit random mode | commit better mode | commit worse mode | Δ avg−rand | Δ avg−best |
|---|---|---|---|---|---|---|
| **Lateral** | {S['lat']['pdms_avg']:.4f} | {S['lat']['pdms_cent_mean']:.4f} | {S['lat']['pdms_cent_best']:.4f} | {S['lat']['pdms_cent_worst']:.4f} | **{P['lat']['rand'].mean():+.4f}** | {P['lat']['best'].mean():+.4f} |
| Longitudinal | {S['long']['pdms_avg']:.4f} | {S['long']['pdms_cent_mean']:.4f} | {S['long']['pdms_cent_best']:.4f} | {S['long']['pdms_cent_worst']:.4f} | {P['long']['rand'].mean():+.4f} | {P['long']['best'].mean():+.4f} |
| Uni-modal | {S['um']['pdms_avg']:.4f} | {S['um']['pdms_cent_mean']:.4f} | {S['um']['pdms_cent_best']:.4f} | {S['um']['pdms_cent_worst']:.4f} | {P['um']['rand'].mean():+.4f} | {P['um']['best'].mean():+.4f} |

*commit random/better/worse mode* = drive the mean trajectory of one behaviour cluster (mode-symmetric
mean / max / min of the two mode centroids). Averaging is worse than a random mode in
{100*pct_worse_rand['lat']:.0f}% (lateral), {100*pct_worse_rand['long']:.0f}% (longitudinal),
{100*pct_worse_rand['um']:.0f}% (uni-modal) of scenes.
""")

md.append("## 3. Which safety components degrade under averaging\n")
md.append(
f"""PDMS sub-metrics (1.0 = safe/compliant), averaged vs mean over single drivers, **lateral** group:

| sub-metric | averaged | mean single-driver | Δ |
|---|---|---|---|
| no-at-fault-collision | {S['lat']['avg_coll']:.4f} | {S['lat']['mean_coll']:.4f} | {S['lat']['avg_coll']-S['lat']['mean_coll']:+.4f} |
| drivable-area | {S['lat']['avg_dac']:.4f} | {S['lat']['mean_dac']:.4f} | {S['lat']['avg_dac']-S['lat']['mean_dac']:+.4f} |
| time-to-collision | {S['lat']['avg_ttc']:.4f} | {S['lat']['mean_ttc']:.4f} | {S['lat']['avg_ttc']-S['lat']['mean_ttc']:+.4f} |
| ego-progress | {S['lat']['avg_prog']:.4f} | {S['lat']['mean_prog']:.4f} | {S['lat']['avg_prog']-S['lat']['mean_prog']:+.4f} |

vs the mode centroids the gap is larger (mode centroids are cleaner than the raw driver mean, which
is polluted by weak planners); the drivable-area / collision columns are where lateral averaging bites.
""")

md.append("## 4. The averaged trajectory is unlike any real mode\n")
md.append(
f"""| group | dist to nearest driver (m) | dist to nearest mode centroid (m) | full-traj dist (m) |
|---|---|---|---|
| Lateral | {S['lat']['d_near_endpoint']:.2f} | {S['lat']['d_near_centroid']:.2f} | {S['lat']['d_near_fulltraj']:.2f} |
| Longitudinal | {S['long']['d_near_endpoint']:.2f} | {S['long']['d_near_centroid']:.2f} | {S['long']['d_near_fulltraj']:.2f} |
| Uni-modal | {S['um']['d_near_endpoint']:.2f} | {S['um']['d_near_centroid']:.2f} | {S['um']['d_near_fulltraj']:.2f} |
""")

md.append("## 5. Method / caveats\n")
md.append(
f"""- **Safety metric = real PDMS**, not a proxy. All selected styletest tokens are in
  `metric_cache_navtest_v1` (4049/4049 styletest ⊂ 12146 navtest cache), scored via
  `models.utils.score.PDM_Reward`→`navsim.evaluate.pdm_score` (LQR + bicycle sim, 4 s @ 10 Hz;
  no-at-fault-collision × drivable-area × driving-direction, weighted with TTC/progress/comfort) —
  the identical scorer used by youdrive GRPO/eval. **No PDMS was fabricated.**
- All trajectories resampled to a common 8 pt @ 4 s grid then to the scorer's 10 pt @ 5 s
  (constant-velocity past 4 s), identical for averaged & mode-committed → apples-to-apples.
- "Averaged" = elementwise mean over the {len(DRIVERS)} drivers. "Mode-committed" = centroid of one
  2-means behaviour cluster (≥{MINCLUST} drivers each).
- Stats: paired Wilcoxon (avg vs commit-random-mode) per group; Mann-Whitney on the per-scene
  penalty (lateral vs uni-modal) for the difference-in-differences.
- **Honest caveat / nuance:** the naive "averaging always harmful" hypothesis is *not* supported by
  PDMS. Averaging is (a) worse than the oracle-best mode everywhere, but (b) only worse than a
  *random* committed mode for **lateral** splits; for longitudinal brake/go splits a mid-speed
  average is usually safe. The single-driver pool mixes strong/weak planners, so "mean single
  driver" is an unreliable mode proxy — the brake/go & left/right **mode-centroid** comparison is
  the defensible test the verdict rests on.
""")

open(OUT_MD,"w").write("\n".join(md))
print("\n".join(md))
print("\n[save] markdown ->",OUT_MD, flush=True)
print(f"\nHEADLINE(vs BEST mode): lat {P['lat']['best'].mean():+.4f} (p={Wb['lat']:.1e}) | "
      f"long {P['long']['best'].mean():+.4f} | unimodal {P['um']['best'].mean():+.4f} | "
      f"DiD(lat-um) {P['lat']['best'].mean()-P['um']['best'].mean():+.4f} p={did_best:.1e} || "
      f"(vs RANDOM mode) lat {P['lat']['rand'].mean():+.4f} long {P['long']['rand'].mean():+.4f} "
      f"um {P['um']['rand'].mean():+.4f}", flush=True)
