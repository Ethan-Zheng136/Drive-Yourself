"""Q1 extension: per-model fail-to-commit (mode-averaging pathology) across ALL 16 models.

Reuses the EXACT scene selection + real-PDMS scorer + commit-vs-middle classifier
from mode_averaging_q1.py / mode_averaging.py:
  * scene sets: 2-means bimodal detection on the 16-driver pool endpoints @4s
    (MIN_DRIVERS=15, MINCLUST=3, GAP_LONG=4.0, GAP_LAT=1.5, SEED=0) -> lateral /
    longitudinal bimodal + unimodal remainder.
  * classifier: nearest of {mode-centroid A, mode-centroid B, pool-average (MID)}
    on the 8pt canonical trajectory (full_dist). "MID" == fail-to-commit == drives
    the non-committal middle. This is IDENTICAL to the human/autovla classifier in
    mode_averaging_q1.py.
  * safety: native nuPlan/navsim PDMS on metric_cache_navtest_v1.

Per-model PDMS is REUSED from mode_averaging_q1_scores.json (pdms_drivers[model] and
pdms_human were computed there by the identical score8 on each model's own traj);
only constant_velocity (absent from that dump) is scored fresh here. Nothing fabricated.
"""
import os, sys, json, time, traceback
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
sys.path.insert(0, AV); sys.path.insert(0, f"{AV}/navsim")
import lzma, pickle
from models.utils.score import PDM_Reward, Trajectory, TrajectorySampling
from navsim.evaluate.pdm_score import pdm_score

CMP = "/mnt/pfs/zhengguantian/autovla/compare_full"
MC = Path("/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
Q1_JSON = "/mnt/pfs/zhengguantian/autovla/persona/mode_averaging_q1_scores.json"
OUT_JSON = "/mnt/pfs/zhengguantian/autovla/persona/per_model_commitment.json"

# --- scene-selection driver pool (IDENTICAL to mode_averaging_q1.py) ---
DRIVERS = ["autovla","diffusiondrive","diffusiondrivev2","drivesuprim","drivoR",
           "goalflow","gtrs","gtrs_aug","gtrs_dp","gtrs_r","hydra_mdp","hydra_mdp_r",
           "sparsedrivev2","trajdiff_scaling","transfuser","wote"]
# --- the 16 models we classify (user request) ---
TARGETS = ["autovla","diffusiondrivev2","diffusiondrive","goalflow","transfuser","gtrs",
           "hydra_mdp","wote","gtrs_dp","gtrs_aug","sparsedrivev2","drivesuprim",
           "trajdiff_scaling","drivoR","constant_velocity","human_logreplay"]
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
# target-model dumps (superset of pool; adds constant_velocity + human_logreplay)
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
    within=np.sqrt(np.mean(np.sum((ep-cc[lab])**2,1)))
    sep=float(np.hypot(*(cc[0]-cc[1]))/(within+1e-6))
    return names,trajs,lab,cc,sizes,sep,long_gap,lat_gap

# --- scene classification (base rate) ---
info={}
for tk,dd in canon_by_tok.items():
    names,trajs,lab,cc,sizes,sep,long_gap,lat_gap=modes_2d(dd)
    minclust=min(sizes); populated=minclust>=MINCLUST
    is_lat = populated and (lat_gap>=GAP_LAT) and (lat_gap>=long_gap)
    is_long= populated and (long_gap>=GAP_LONG) and (long_gap>lat_gap)
    info[tk]=dict(is_lat=is_lat,is_long=is_long,minclust=minclust)

lat_toks =[t for t,v in info.items() if v["is_lat"]]
long_toks=[t for t,v in info.items() if v["is_long"]]
n_tokens=len(canon_by_tok); n_lat=len(lat_toks); n_long=len(long_toks)
n_uni=n_tokens-n_lat-n_long
print(f"[base-rate] tokens={n_tokens} lateral={n_lat} ({100*n_lat/n_tokens:.2f}%) "
      f"longitudinal={n_long} ({100*n_long/n_tokens:.2f}%) unimodal={n_uni} ({100*n_uni/n_tokens:.2f}%)", flush=True)

# --- classifier: nearest of {centA, centB, avg(MID)} on 8pt traj (== human classifier) ---
def classify(traj, cents, avg):
    dists={"A":full_dist(traj,cents[0]),"B":full_dist(traj,cents[1]),"MID":full_dist(traj,avg)}
    return min(dists,key=dists.get)

def target_traj(d, tk):
    if tk in traw[d]:
        return canon(traw[d][tk], float(tmeta[d]["horizon_s"]), int(tmeta[d]["n_pts"]))
    return None

# per token & model classification over bimodal scenes
cls={"lat":{}, "long":{}}          # cls[group][tok] = {model: A/B/MID, "_human": class}
selfcheck={"human_match":0,"human_tot":0,"autovla_match":0,"autovla_tot":0}
q1=json.load(open(Q1_JSON))
q1map={"lat":{r["tok"]:r for r in q1["lat"]}, "long":{r["tok"]:r for r in q1["long"]}}

for group,toks in (("lat",lat_toks),("long",long_toks)):
    for tk in toks:
        dd=canon_by_tok[tk]
        names,trajs,lab,cc,sizes,sep,long_gap,lat_gap=modes_2d(dd)
        avg=trajs.mean(0)
        cents=[trajs[lab==0].mean(0), trajs[lab==1].mean(0)]
        rec={}
        for d in TARGETS:
            tr=target_traj(d,tk)
            rec[d]= classify(tr,cents,avg) if tr is not None else None
        cls[group][tk]=rec
        # self-check vs q1 stored human_class / autovla_class
        q=q1map[group].get(tk)
        if q is not None:
            if q.get("human_class") is not None and rec["human_logreplay"] is not None:
                selfcheck["human_tot"]+=1; selfcheck["human_match"]+= (q["human_class"]==rec["human_logreplay"])
            if q.get("autovla_class") is not None and rec["autovla"] is not None:
                selfcheck["autovla_tot"]+=1; selfcheck["autovla_match"]+= (q["autovla_class"]==rec["autovla"])
print(f"[selfcheck] human_class match {selfcheck['human_match']}/{selfcheck['human_tot']} "
      f"autovla_class match {selfcheck['autovla_match']}/{selfcheck['autovla_tot']} "
      "(vs mode_averaging_q1_scores.json)", flush=True)

# --- PDMS: reuse q1 (pdms_drivers / pdms_human); score constant_velocity fresh ---
def score8(xy8,tok):
    xy10=to_score10(xy8); hs=headings(xy10)
    traj=Trajectory(np.concatenate([xy10,hs[:,None]],1),samp)
    with lzma.open(rew.metric_cache_loader.metric_cache_paths[tok],"rb") as f: mc=pickle.load(f)
    r=pdm_score(metric_cache=mc,model_trajectory=traj,future_sampling=rew.future_sampling,
                simulator=rew.simulator,scorer=rew.scorer)
    return float(r.score)

pdms={"lat":{}, "long":{}}          # pdms[group][tok] = {model: score}
cv_t0=time.time(); cv_n=0
for group,toks in (("lat",lat_toks),("long",long_toks)):
    for tk in toks:
        q=q1map[group][tk]
        rec={}
        for d in TARGETS:
            if d=="human_logreplay":
                rec[d]=q.get("pdms_human")
            elif d=="constant_velocity":
                tr=target_traj(d,tk)
                rec[d]=score8(tr,tk) if tr is not None else None
                cv_n+=1
                if cv_n%50==0: print(f"  [cv-score] {cv_n} {time.time()-cv_t0:.0f}s", flush=True)
            else:
                rec[d]=q["pdms_drivers"].get(d)
        pdms[group][tk]=rec
print(f"[pdms] constant_velocity scored {cv_n} tokens in {time.time()-cv_t0:.0f}s; others reused from q1", flush=True)

# ---------------- aggregate ----------------
def middle_rate(group,d):
    v=[cls[group][tk][d] for tk in cls[group] if cls[group][tk][d] is not None]
    n=len(v); mid=sum(x=="MID" for x in v)
    return (mid/n if n else float("nan")), mid, n

def commit_rate(group,d):  # committed (A or B) fraction
    v=[cls[group][tk][d] for tk in cls[group] if cls[group][tk][d] is not None]
    n=len(v); com=sum(x in ("A","B") for x in v)
    return (com/n if n else float("nan")), com, n

# divergence from human: scenes where human committed
def divergence(d):
    same=other=mid=tot=0
    for group in ("lat","long"):
        for tk in cls[group]:
            h=cls[group][tk]["human_logreplay"]; m=cls[group][tk][d]
            if h in ("A","B") and m is not None:
                tot+=1
                if m=="MID": mid+=1
                elif m==h: same+=1
                else: other+=1
    return dict(same=same,other=other,mid=mid,tot=tot,
                same_r=same/tot if tot else float("nan"),
                other_r=other/tot if tot else float("nan"),
                mid_r=mid/tot if tot else float("nan"))

def mean_pdms(group,d):
    v=[pdms[group][tk][d] for tk in pdms[group] if pdms[group][tk][d] is not None]
    return (float(np.mean(v)) if v else float("nan")), len(v)

agg={}
for d in TARGETS:
    mlat,mlat_n,nlat=middle_rate("lat",d)
    mlong,mlong_n,nlong=middle_rate("long",d)
    # combined middle-rate over lat+long
    allv=[cls[g][tk][d] for g in ("lat","long") for tk in cls[g] if cls[g][tk][d] is not None]
    mall=sum(x=="MID" for x in allv)/len(allv) if allv else float("nan")
    plat,plat_n=mean_pdms("lat",d); plong,plong_n=mean_pdms("long",d)
    pall_v=[pdms[g][tk][d] for g in ("lat","long") for tk in pdms[g] if pdms[g][tk][d] is not None]
    pall=float(np.mean(pall_v)) if pall_v else float("nan")
    agg[d]=dict(mid_lat=mlat,mid_lat_n=nlat,mid_long=mlong,mid_long_n=nlong,mid_all=mall,
                pdms_lat=plat,pdms_long=plong,pdms_all=pall,
                div=divergence(d))

# correlation middle-rate(all) vs pdms(all) across models
mm=np.array([agg[d]["mid_all"] for d in TARGETS])
pp=np.array([agg[d]["pdms_all"] for d in TARGETS])
ok=~(np.isnan(mm)|np.isnan(pp))
try:
    from scipy import stats
    pear=stats.pearsonr(mm[ok],pp[ok]); spear=stats.spearmanr(mm[ok],pp[ok])
    corr=dict(pearson_r=float(pear[0]),pearson_p=float(pear[1]),
              spearman_r=float(spear[0]),spearman_p=float(spear[1]),n=int(ok.sum()))
except Exception as e:
    corr=dict(error=repr(e))
print(f"[corr] middle-rate vs PDMS  pearson r={corr.get('pearson_r'):.3f} p={corr.get('pearson_p'):.2g} "
      f"spearman rho={corr.get('spearman_r'):.3f} p={corr.get('spearman_p'):.2g}", flush=True)

out=dict(base_rate=dict(n_tokens=n_tokens,n_lat=n_lat,n_long=n_long,n_uni=n_uni,
             pct_lat=100*n_lat/n_tokens,pct_long=100*n_long/n_tokens,pct_uni=100*n_uni/n_tokens),
         selfcheck=selfcheck, agg=agg, corr=corr,
         config=dict(DRIVERS=DRIVERS,TARGETS=TARGETS,MIN_DRIVERS=MIN_DRIVERS,MINCLUST=MINCLUST,
                     GAP_LONG=GAP_LONG,GAP_LAT=GAP_LAT,SEED=SEED,n_lat=n_lat,n_long=n_long))
json.dump(out,open(OUT_JSON,"w"),indent=1)
print("[save] ->",OUT_JSON, flush=True)
print("DONE", flush=True)
