"""Safety link for "models diverge from human commitment".

WITHIN the 652 scenes where the HUMAN committed to a maneuver mode, and
WITHIN each model, compare the model's OWN real PDMS on scenes where it
MATCHED the human's committed mode vs scenes where it DIVERGED (wrong real
mode OR non-committal middle). This is a within-model, within-human-committed
comparison -- it does NOT use the (confounded) cross-model middle-rate<->PDMS
correlation.

Reuses EXACTLY the scene selection + 3-way commit classifier + real-PDMS
scorer from per_model_commitment.py / mode_averaging_q1.py:
  * bimodal scene detection: 2-means on 16-driver endpoints @4s
    (MIN_DRIVERS=15, MINCLUST=3, GAP_LONG=4.0, GAP_LAT=1.5, SEED=0)
  * per-(model,scene) class: nearest of {mode A, mode B, pool-avg MID} on the
    8pt canonical trajectory (full_dist) -- IDENTICAL classifier for every model
  * PDMS: native navsim pdm_score on metric_cache_navtest_v1. Per-model PDMS
    reused from mode_averaging_q1_scores.json (pdms_drivers / pdms_human);
    constant_velocity scored fresh with the same PDM_Reward->pdm_score path.
Nothing fabricated; real PDMS only.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
sys.path.insert(0, AV); sys.path.insert(0, f"{AV}/navsim")
import lzma, pickle
from models.utils.score import PDM_Reward, Trajectory, TrajectorySampling
from navsim.evaluate.pdm_score import pdm_score
from scipy import stats

CMP = "/mnt/pfs/zhengguantian/autovla/compare_full"
MC = Path("/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
Q1_JSON = "/mnt/pfs/zhengguantian/autovla/persona/mode_averaging_q1_scores.json"
PERSONA = "/mnt/pfs/zhengguantian/autovla/persona"
OUT_JSON = f"{PERSONA}/divergence_safety_link.json"
OUT_MD = f"{PERSONA}/divergence_safety_link.md"
OUT_PNG = f"{PERSONA}/divergence_safety_link.png"

DRIVERS = ["autovla","diffusiondrive","diffusiondrivev2","drivesuprim","drivoR",
           "goalflow","gtrs","gtrs_aug","gtrs_dp","gtrs_r","hydra_mdp","hydra_mdp_r",
           "sparsedrivev2","trajdiff_scaling","transfuser","wote"]
TARGETS = ["autovla","diffusiondrivev2","diffusiondrive","goalflow","transfuser","gtrs",
           "hydra_mdp","wote","gtrs_dp","gtrs_aug","sparsedrivev2","drivesuprim",
           "trajdiff_scaling","drivoR","constant_velocity","human_logreplay"]
HUMAN = "human_logreplay"
NONHUMAN = [d for d in TARGETS if d != HUMAN]  # 15 (incl. constant_velocity)
MIN_DRIVERS=15; MINCLUST=3; GAP_LONG=4.0; GAP_LAT=1.5; SEED=0
t_common=np.linspace(0.5,4.0,8); t_dst=np.linspace(0.5,5.0,10)

def canon(xy,H,n):
    xy=np.asarray(xy,float); t_src=np.linspace(H/n,H,n)
    v=(xy[-1]-xy[-2])/(t_src[-1]-t_src[-2]); out=np.zeros((8,2))
    for i,t in enumerate(t_common):
        if t<=t_src[-1]+1e-6:
            out[i,0]=np.interp(t,t_src,xy[:,0]); out[i,1]=np.interp(t,t_src,xy[:,1])
        else: out[i]=xy[-1]+v*(t-t_src[-1])
    return out
def to_score10(xy8):
    v=(xy8[-1]-xy8[-2])/(t_common[-1]-t_common[-2]); out=np.zeros((10,2))
    for i,t in enumerate(t_dst):
        if t<=t_common[-1]+1e-6:
            out[i,0]=np.interp(t,t_common,xy8[:,0]); out[i,1]=np.interp(t,t_common,xy8[:,1])
        else: out[i]=xy8[-1]+v*(t-t_common[-1])
    return out
def headings(xy10):
    prev=np.array([0.,0.]); hs=[]
    for p in xy10:
        d=p-prev; hs.append(np.arctan2(d[1],d[0])); prev=p
    return np.array(hs)
def full_dist(a,b):
    return float(np.mean(np.hypot(a[:,0]-b[:,0], a[:,1]-b[:,1])))

print("[load] reading driver pool dumps ...", flush=True)
raw={d:json.load(open(f"{CMP}/{d}.json")) for d in DRIVERS}
meta={d:raw[d]["_meta"] for d in DRIVERS}
traw={}; tmeta={}
for d in TARGETS:
    if d in raw: traw[d]=raw[d]; tmeta[d]=meta[d]
    else:
        traw[d]=json.load(open(f"{CMP}/{d}.json")); tmeta[d]=traw[d]["_meta"]

rew=PDM_Reward(MC)
avail=set(rew.metric_cache_loader.metric_cache_paths.keys())
samp=TrajectorySampling(num_poses=10,interval_length=0.5)

all_toks=set()
for d in DRIVERS: all_toks|=(set(raw[d])-{"_meta"})
all_toks&=avail
canon_by_tok={}
for tk in all_toks:
    dd={d:canon(raw[d][tk],float(meta[d]["horizon_s"]),int(meta[d]["n_pts"])) for d in DRIVERS if tk in raw[d]}
    if len(dd)>=MIN_DRIVERS:
        canon_by_tok[tk]=dd
print(f"[load] {len(canon_by_tok)} tokens >= {MIN_DRIVERS} drivers (of {len(all_toks)} in cache)", flush=True)

def modes_2d(dd):
    names=sorted(dd); trajs=np.stack([dd[d] for d in names]); ep=trajs[:,-1,:]
    km=KMeans(2,n_init=4,random_state=SEED).fit(ep); lab=km.labels_; cc=km.cluster_centers_
    sizes=[int((lab==0).sum()),int((lab==1).sum())]
    long_gap=float(abs(cc[0,0]-cc[1,0])); lat_gap=float(abs(cc[0,1]-cc[1,1]))
    return names,trajs,lab,cc,sizes,long_gap,lat_gap

info={}
for tk,dd in canon_by_tok.items():
    names,trajs,lab,cc,sizes,long_gap,lat_gap=modes_2d(dd)
    minclust=min(sizes); populated=minclust>=MINCLUST
    is_lat = populated and (lat_gap>=GAP_LAT) and (lat_gap>=long_gap)
    is_long= populated and (long_gap>=GAP_LONG) and (long_gap>lat_gap)
    info[tk]=dict(is_lat=is_lat,is_long=is_long)
lat_toks =[t for t,v in info.items() if v["is_lat"]]
long_toks=[t for t,v in info.items() if v["is_long"]]
print(f"[base-rate] lateral={len(lat_toks)} longitudinal={len(long_toks)}", flush=True)

def classify(traj, cents, avg):
    dists={"A":full_dist(traj,cents[0]),"B":full_dist(traj,cents[1]),"MID":full_dist(traj,avg)}
    return min(dists,key=dists.get)
def target_traj(d, tk):
    if tk in traw[d]:
        return canon(traw[d][tk], float(tmeta[d]["horizon_s"]), int(tmeta[d]["n_pts"]))
    return None

cls={"lat":{}, "long":{}}
for group,toks in (("lat",lat_toks),("long",long_toks)):
    for tk in toks:
        dd=canon_by_tok[tk]
        names,trajs,lab,cc,sizes,long_gap,lat_gap=modes_2d(dd)
        avg=trajs.mean(0)
        cents=[trajs[lab==0].mean(0), trajs[lab==1].mean(0)]
        rec={}
        for d in TARGETS:
            tr=target_traj(d,tk)
            rec[d]= classify(tr,cents,avg) if tr is not None else None
        cls[group][tk]=rec

# --- PDMS per (model,scene): reuse q1; score constant_velocity fresh ---
def score8(xy8,tok):
    xy10=to_score10(xy8); hs=headings(xy10)
    traj=Trajectory(np.concatenate([xy10,hs[:,None]],1),samp)
    with lzma.open(rew.metric_cache_loader.metric_cache_paths[tok],"rb") as f: mc=pickle.load(f)
    r=pdm_score(metric_cache=mc,model_trajectory=traj,future_sampling=rew.future_sampling,
                simulator=rew.simulator,scorer=rew.scorer)
    return float(r.score)

q1=json.load(open(Q1_JSON))
q1map={"lat":{r["tok"]:r for r in q1["lat"]}, "long":{r["tok"]:r for r in q1["long"]}}
pdms={"lat":{}, "long":{}}
cv_t0=time.time(); cv_n=0
for group,toks in (("lat",lat_toks),("long",long_toks)):
    for tk in toks:
        q=q1map[group][tk]
        rec={}
        for d in TARGETS:
            if d==HUMAN:
                rec[d]=q.get("pdms_human")
            elif d=="constant_velocity":
                tr=target_traj(d,tk)
                rec[d]=score8(tr,tk) if tr is not None else None
                cv_n+=1
                if cv_n%100==0: print(f"  [cv-score] {cv_n} {time.time()-cv_t0:.0f}s", flush=True)
            else:
                rec[d]=q["pdms_drivers"].get(d)
        pdms[group][tk]=rec
print(f"[pdms] constant_velocity scored {cv_n} tokens in {time.time()-cv_t0:.0f}s; others reused", flush=True)

# ================= PARTITION ANALYSIS =================
# Human-committed scenes = human class in (A,B). For each model partition those
# scenes into matched / wrong-mode / middle, collect the MODEL'S OWN PDMS.
def gather(d):
    matched=[]; wrong=[]; middle=[]
    for group in ("lat","long"):
        for tk in cls[group]:
            h=cls[group][tk][HUMAN]; m=cls[group][tk][d]
            p=pdms[group][tk][d]
            if h in ("A","B") and m is not None and p is not None:
                if m=="MID": middle.append(p)
                elif m==h:   matched.append(p)
                else:        wrong.append(p)
    return matched, wrong, middle

def mean(x): return float(np.mean(x)) if len(x) else float("nan")

def mwu(a,b):
    if len(a)>=5 and len(b)>=5:
        try:
            u,p=stats.mannwhitneyu(a,b,alternative="two-sided")
            return float(p)
        except Exception: return None
    return None

per_model={}
for d in NONHUMAN:
    matched,wrong,middle=gather(d)
    diverged=wrong+middle
    per_model[d]=dict(
        n_matched=len(matched), n_wrong=len(wrong), n_middle=len(middle), n_diverged=len(diverged),
        pdms_matched=mean(matched), pdms_wrong=mean(wrong), pdms_middle=mean(middle),
        pdms_diverged=mean(diverged),
        delta_div_minus_matched=(mean(diverged)-mean(matched)) if matched and diverged else float("nan"),
        p_matched_vs_diverged=mwu(matched,diverged),
    )

# ---- Human reference: human PDMS on the 652 human-committed scenes ----
human_pdms=[]
n_hc=0
for group in ("lat","long"):
    for tk in cls[group]:
        if cls[group][tk][HUMAN] in ("A","B"):
            n_hc+=1
            p=pdms[group][tk][HUMAN]
            if p is not None: human_pdms.append(p)
human_ref=dict(n_committed_scenes=n_hc, n_pdms=len(human_pdms), pdms_mean=mean(human_pdms))
print(f"[human] committed scenes={n_hc}  mean PDMS={mean(human_pdms):.4f}", flush=True)

# ---- POOLED across all non-human models (observation-level; headline) ----
def pool(field):
    M=[]; W=[]; Mi=[]
    for d in NONHUMAN:
        m,w,mi=gather(d)
        M+=m; W+=w; Mi+=mi
    return M,W,Mi
P_matched,P_wrong,P_middle=pool(None)
P_diverged=P_wrong+P_middle
pooled=dict(
    n_matched=len(P_matched), n_wrong=len(P_wrong), n_middle=len(P_middle), n_diverged=len(P_diverged),
    pdms_matched=mean(P_matched), pdms_wrong=mean(P_wrong), pdms_middle=mean(P_middle),
    pdms_diverged=mean(P_diverged),
    delta_div_minus_matched=mean(P_diverged)-mean(P_matched),
    p_matched_vs_diverged=mwu(P_matched,P_diverged),
    p_matched_vs_wrong=mwu(P_matched,P_wrong),
    p_matched_vs_middle=mwu(P_matched,P_middle),
    p_wrong_vs_middle=mwu(P_wrong,P_middle),
)

# pooled excluding constant_velocity (degenerate baseline) -- robustness
def pool_excl_cv():
    M=[]; D=[]
    for d in NONHUMAN:
        if d=="constant_velocity": continue
        m,w,mi=gather(d); M+=m; D+=w+mi
    return M,D
Me,De=pool_excl_cv()
pooled_excl_cv=dict(n_matched=len(Me),n_diverged=len(De),pdms_matched=mean(Me),
                    pdms_diverged=mean(De),delta_div_minus_matched=mean(De)-mean(Me),
                    p_matched_vs_diverged=mwu(Me,De))

# ---- WITHIN-MODEL PAIRED test (confound-robust): per-model means -> Wilcoxon ----
def paired(models):
    ma=[]; di=[]
    for d in models:
        r=per_model[d]
        if r["n_matched"]>=5 and r["n_diverged"]>=5:
            ma.append(r["pdms_matched"]); di.append(r["pdms_diverged"])
    ma=np.array(ma); di=np.array(di)
    out=dict(n_models=len(ma), mean_matched=float(ma.mean()) if len(ma) else float("nan"),
             mean_diverged=float(di.mean()) if len(di) else float("nan"),
             mean_delta=float((di-ma).mean()) if len(ma) else float("nan"),
             n_delta_negative=int((di<ma).sum()))
    if len(ma)>=6:
        try:
            w,p=stats.wilcoxon(ma,di); out["wilcoxon_p"]=float(p)
        except Exception as e: out["wilcoxon_p"]=None
    else: out["wilcoxon_p"]=None
    return out
paired_all=paired(NONHUMAN)
paired_excl_cv=paired([d for d in NONHUMAN if d!="constant_velocity"])

result=dict(config=dict(DRIVERS=DRIVERS,TARGETS=TARGETS,NONHUMAN=NONHUMAN,
                        MIN_DRIVERS=MIN_DRIVERS,MINCLUST=MINCLUST,GAP_LONG=GAP_LONG,
                        GAP_LAT=GAP_LAT,SEED=SEED,n_lat=len(lat_toks),n_long=len(long_toks)),
            human_committed_scenes=n_hc, human_ref=human_ref,
            per_model=per_model, pooled=pooled, pooled_excl_cv=pooled_excl_cv,
            paired_all=paired_all, paired_excl_cv=paired_excl_cv)
json.dump(result, open(OUT_JSON,"w"), indent=1)
print("[save] ->", OUT_JSON, flush=True)

# ================= FIGURE =================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.2,5.0))
cats=["Matched\nhuman mode","Wrong\nmode","Middle\n(no commit)"]
vals=[pooled["pdms_matched"],pooled["pdms_wrong"],pooled["pdms_middle"]]
ns=[pooled["n_matched"],pooled["n_wrong"],pooled["n_middle"]]
colors=["#2e7d32","#c62828","#ef6c00"]
bars=ax.bar(cats,vals,color=colors,width=0.62,edgecolor="black",linewidth=0.6)
for b,v,n in zip(bars,vals,ns):
    ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}\n(n={n})",
            ha="center",va="bottom",fontsize=10)
hy=human_ref["pdms_mean"]
ax.axhline(hy,ls="--",color="#1565c0",lw=1.6)
ax.text(2.48,hy+0.004,f"Human ref = {hy:.3f}",color="#1565c0",ha="right",va="bottom",fontsize=10)
ax.set_ylabel("Model's own mean real PDMS")
ax.set_title("Within human-committed scenes: PDMS by model's own behavior\n"
             f"(pooled over {len(NONHUMAN)} non-human models, {n_hc} human-committed scenes)")
ax.set_ylim(0, max(max(vals),hy)*1.16)
ax.grid(axis="y",ls=":",alpha=0.5)
d=pooled["delta_div_minus_matched"]; p=pooled["p_matched_vs_diverged"]
ax.text(0.02,0.02,f"Diverged - Matched Δ = {d:+.3f}  (Mann-Whitney p = {p:.2e})",
        transform=ax.transAxes,fontsize=9.5,style="italic")
fig.tight_layout()
fig.savefig(OUT_PNG,dpi=120)
print("[save] ->", OUT_PNG, flush=True)
print("POOLED matched=%.4f diverged=%.4f delta=%.4f p=%s" % (
    pooled["pdms_matched"],pooled["pdms_diverged"],pooled["delta_div_minus_matched"],
    pooled["p_matched_vs_diverged"]), flush=True)
print("DONE", flush=True)
