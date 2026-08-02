"""Q1: In multi-modal (esp. lateral left/right) scenes where the AVERAGED trajectory
is penalized, is STRAIGHT/averaging genuinely the wrong/less-safe choice, or is
straight a fine answer we're just comparing to cherry-picked better modes?

Reuses mode_averaging.py scene-selection + real PDMS scoring, extends with the
HUMAN ground-truth (human_logreplay) oracle and AutoVLA's actual output.

Safety metric: native nuPlan/navsim PDMS on metric_cache_navtest_v1 (all 4049
styletest tokens present). Every number from the real scorer; nothing fabricated.
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
OUT_JSON = "/mnt/pfs/zhengguantian/autovla/persona/mode_averaging_q1_scores.json"
DRIVERS = ["autovla","diffusiondrive","diffusiondrivev2","drivesuprim","drivoR",
           "goalflow","gtrs","gtrs_aug","gtrs_dp","gtrs_r","hydra_mdp","hydra_mdp_r",
           "sparsedrivev2","trajdiff_scaling","transfuser","wote"]
HUMAN = "human_logreplay"
MIN_DRIVERS=15; MINCLUST=3; GAP_LONG=4.0; GAP_LAT=1.5; SEED=0
UM_K = int(os.environ.get("UM_K","300"))   # unimodal control size
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
def full_dist(a,b):  # mean per-point euclidean over 8pt canon trajs
    return float(np.mean(np.hypot(a[:,0]-b[:,0], a[:,1]-b[:,1])))
def geom(xy8):
    """commitment descriptors on an 8pt @4s ego traj (x fwd, y left)."""
    ep=xy8[-1]
    lat=float(ep[1])                                   # signed lateral disp @4s (+left)
    # net heading change: final segment direction relative to +x
    seg=xy8[-1]-xy8[-2]; net_head=float(np.arctan2(seg[1],seg[0]))
    # integrated |curvature| proxy: sum of |turn angle| between consecutive segments
    pts=np.vstack([[0,0],xy8]); segs=np.diff(pts,axis=0)
    ang=np.arctan2(segs[:,1],segs[:,0]); dang=np.diff(ang)
    dang=(dang+np.pi)%(2*np.pi)-np.pi
    tot_turn=float(np.sum(np.abs(dang)))
    return dict(lat=lat, abs_lat=abs(lat), net_head=net_head, tot_turn=tot_turn)

print("[load] reading driver dumps + human GT ...", flush=True)
raw={d:json.load(open(f"{CMP}/{d}.json")) for d in DRIVERS}
meta={d:raw[d]["_meta"] for d in DRIVERS}
hraw=json.load(open(f"{CMP}/{HUMAN}.json")); hmeta=hraw["_meta"]
rew=PDM_Reward(MC)
avail=set(rew.metric_cache_loader.metric_cache_paths.keys())
samp=TrajectorySampling(num_poses=10,interval_length=0.5)

all_toks=set()
for d in DRIVERS: all_toks|=(set(raw[d])-{"_meta"})
all_toks&=avail
canon_by_tok={}; human_by_tok={}
for tk in all_toks:
    dd={d:canon(raw[d][tk],float(meta[d]["horizon_s"]),int(meta[d]["n_pts"])) for d in DRIVERS if tk in raw[d]}
    if len(dd)>=MIN_DRIVERS:
        canon_by_tok[tk]=dd
        if tk in hraw: human_by_tok[tk]=canon(hraw[tk],float(hmeta["horizon_s"]),int(hmeta["n_pts"]))
print(f"[load] {len(canon_by_tok)} tokens >= {MIN_DRIVERS} drivers; human GT for {len(human_by_tok)}", flush=True)

def modes_2d(dd):
    names=sorted(dd); trajs=np.stack([dd[d] for d in names]); ep=trajs[:,-1,:]
    km=KMeans(2,n_init=4,random_state=SEED).fit(ep); lab=km.labels_; cc=km.cluster_centers_
    sizes=[int((lab==0).sum()),int((lab==1).sum())]
    long_gap=float(abs(cc[0,0]-cc[1,0])); lat_gap=float(abs(cc[0,1]-cc[1,1]))
    within=np.sqrt(np.mean(np.sum((ep-cc[lab])**2,1)))
    sep=float(np.hypot(*(cc[0]-cc[1]))/(within+1e-6))
    return names,trajs,lab,cc,sizes,sep,long_gap,lat_gap
def pairwise_mean(pts):
    n=len(pts); s=0.; c=0
    for i in range(n):
        for j in range(i+1,n): s+=np.hypot(*(pts[i]-pts[j])); c+=1
    return s/max(c,1)

info={}
for tk,dd in canon_by_tok.items():
    names,trajs,lab,cc,sizes,sep,long_gap,lat_gap=modes_2d(dd)
    ep=trajs[:,-1,:]; minclust=min(sizes)
    populated=minclust>=MINCLUST
    is_lat = populated and (lat_gap>=GAP_LAT) and (lat_gap>=long_gap)
    is_long= populated and (long_gap>=GAP_LONG) and (long_gap>lat_gap)
    info[tk]=dict(disagree=pairwise_mean(ep),sep=sep,long_gap=long_gap,lat_gap=lat_gap,
                  sizes=sizes,minclust=minclust,is_lat=is_lat,is_long=is_long)

lat_toks =[t for t,v in info.items() if v["is_lat"]]
long_toks=[t for t,v in info.items() if v["is_long"]]
um_toks  =sorted(info,key=lambda t:info[t]["disagree"])[:UM_K]
print(f"[select] lateral={len(lat_toks)} longitudinal={len(long_toks)} unimodal_ctrl={len(um_toks)}", flush=True)

def score8(xy8,tok):
    xy10=to_score10(xy8); hs=headings(xy10)
    traj=Trajectory(np.concatenate([xy10,hs[:,None]],1),samp)
    with lzma.open(rew.metric_cache_loader.metric_cache_paths[tok],"rb") as f: mc=pickle.load(f)
    r=pdm_score(metric_cache=mc,model_trajectory=traj,future_sampling=rew.future_sampling,
                simulator=rew.simulator,scorer=rew.scorer)
    return dict(score=float(r.score),coll=float(r.no_at_fault_collisions),
                dac=float(r.drivable_area_compliance),prog=float(r.ego_progress),
                ttc=float(r.time_to_collision_within_bound),comfort=float(r.comfort),
                ddc=float(r.driving_direction_compliance))

def analyze(tokens,label):
    out=[]; t0=time.time(); nerr=0
    for i,tk in enumerate(tokens):
        if i%25==0:
            print(f"  [{label}] {i}/{len(tokens)} {time.time()-t0:.0f}s", flush=True)
        dd=canon_by_tok[tk]
        names,trajs,lab,cc,sizes,sep,long_gap,lat_gap=modes_2d(dd)
        try:
            drv_scores={d:score8(dd[d],tk) for d in names}
            avg=trajs.mean(0); s_avg=score8(avg,tk)
            # mode centroids (within-mode averages)
            cents=[trajs[lab==c].mean(0) for c in range(2)]
            s_cent=[score8(cents[c],tk) for c in range(2)]
            # human GT
            has_h = tk in human_by_tok
            h=human_by_tok.get(tk); s_h=score8(h,tk) if has_h else None
            # autovla actual (already scored as a driver)
            s_av=drv_scores["autovla"]; av_traj=dd["autovla"]
            # --- geometry / commitment ---
            g_avg=geom(avg); g_c=[geom(cents[0]),geom(cents[1])]; g_h=geom(h) if has_h else None
            g_av=geom(av_traj)
            # --- distances for classification ---
            d_h=None; h_class=None; d_h_avg=None; d_h_nearcent=None
            if has_h:
                d_h_cA=full_dist(h,cents[0]); d_h_cB=full_dist(h,cents[1]); d_h_avg=full_dist(h,avg)
                dists={"A":d_h_cA,"B":d_h_cB,"MID":d_h_avg}
                h_class=min(dists,key=dists.get)   # nearest of {centA,centB,avg}
                d_h_nearcent=float(min(d_h_cA,d_h_cB))
                d_h={"cA":d_h_cA,"cB":d_h_cB,"avg":d_h_avg}
            # autovla classification (nearest of centA/centB/avg)
            av_dists={"A":full_dist(av_traj,cents[0]),"B":full_dist(av_traj,cents[1]),"MID":full_dist(av_traj,avg)}
            av_class=min(av_dists,key=av_dists.get)
            av_d_avg=av_dists["MID"]; av_d_nearcent=float(min(av_dists["A"],av_dists["B"]))
            # which centroid endpoint is "left" (higher y)
            left_is0 = cents[0][-1,1] >= cents[1][-1,1]
            pv=np.array([drv_scores[d]["score"] for d in names])
            cent_sc=np.array([s_cent[0]["score"],s_cent[1]["score"]])
            eps=trajs[:,-1,:]; ep_avg=avg[-1]
            d_near_drv=float(np.min(np.hypot(eps[:,0]-ep_avg[0],eps[:,1]-ep_avg[1])))
            rec=dict(tok=tk,label=label,ndrv=len(names),sizes=sizes,
                info=dict(disagree=info[tk]["disagree"],sep=sep,long_gap=long_gap,lat_gap=lat_gap,minclust=info[tk]["minclust"]),
                # PDMS
                pdms_avg=s_avg["score"], pdms_cA=cent_sc[0], pdms_cB=cent_sc[1],
                pdms_cent_best=float(cent_sc.max()), pdms_cent_worst=float(cent_sc.min()),
                pdms_cent_mean=float(cent_sc.mean()),
                pdms_human=(s_h["score"] if has_h else None),
                pdms_autovla=s_av["score"],
                pdms_drivers={d:drv_scores[d]["score"] for d in names},
                pdms_drv_mean=float(pv.mean()), pdms_drv_med=float(np.median(pv)),
                pdms_drv_best=float(pv.max()), pdms_drv_worst=float(pv.min()),
                avg_below_both=bool(s_avg["score"] < cent_sc.min()-1e-6),
                avg_below_human=(bool(s_avg["score"] < s_h["score"]-1e-6) if has_h else None),
                # submetrics
                sub_avg=s_avg, sub_cA=s_cent[0], sub_cB=s_cent[1],
                sub_human=(s_h if has_h else None), sub_autovla=s_av,
                # geometry
                geom_avg=g_avg, geom_cA=g_c[0], geom_cB=g_c[1],
                geom_human=(g_h if has_h else None), geom_autovla=g_av,
                left_is0=bool(left_is0),
                # classification / distances
                has_human=has_h, human_class=h_class, human_dists=d_h,
                d_human_avg=d_h_avg, d_human_nearcent=d_h_nearcent,
                autovla_class=av_class, d_autovla_avg=av_d_avg, d_autovla_nearcent=av_d_nearcent,
                d_avg_nearest_driver=d_near_drv,
            )
            out.append(rec)
        except Exception as e:
            nerr+=1; print(f"  [{label}] ERROR {tk}: {repr(e)[:140]}", flush=True); traceback.print_exc()
    print(f"  [{label}] done {len(out)} ok {nerr} err {time.time()-t0:.0f}s", flush=True)
    return out

print("[score] LATERAL ...", flush=True); res_lat=analyze(lat_toks,"lat")
print("[score] LONGITUDINAL ...", flush=True); res_long=analyze(long_toks,"long")
print("[score] UNIMODAL ctrl ...", flush=True); res_um=analyze(um_toks,"um")

json.dump({"lat":res_lat,"long":res_long,"um":res_um,
    "config":dict(MIN_DRIVERS=MIN_DRIVERS,MINCLUST=MINCLUST,GAP_LONG=GAP_LONG,GAP_LAT=GAP_LAT,
        UM_K=UM_K,drivers=DRIVERS,n_lat=len(lat_toks),n_long=len(long_toks),
        n_tokens=len(canon_by_tok),n_human=len(human_by_tok))},
    open(OUT_JSON,"w"))
print("[save] ->",OUT_JSON, flush=True)
print("DONE", flush=True)
