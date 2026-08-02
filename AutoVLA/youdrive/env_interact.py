"""Environment-interaction style features (group B).

For each scene, loads GT agents (other vehicles + VRUs) over the future horizon,
transforms them into the current (t=0) ego frame, and computes interaction
features between EACH model's predicted ego trajectory and the real agents:

  thw_min      min time-headway to lead vehicle      (small = aggressive following)
  ttc_min      min time-to-collision to lead         (small = aggressive)
  dist_any_min min distance to ANY agent             (small = aggressive proximity)
  dist_vru_min min distance to pedestrian/cyclist    (small = aggressive near VRU)
  lead_gap_min min longitudinal gap to lead (m)

Runs once over scenes; scores human GT + all model dumps in the same pass.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

DATA="/root/workspace/closed_loop/data/navsim"
CMP="/mnt/pfs/zhengguantian/autovla/compare_full"
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
DT=0.5
CORRIDOR=1.5           # lateral half-width (m) to count an agent as "in my lane / lead"
VRU_KEYS=("pedestrian","bicycle","cyclist","bike")
OUT=os.environ.get("OUT","youdrive/env_interact_out.json")
CAP=int(os.environ.get("CAP","99999"))

_only=os.environ.get("ONLY_MODELS")  # json: {"name":"/path.json", ...}
if _only:
    MODELS={n:json.load(open(p)) for n,p in json.loads(_only).items()}
else:
    MODELS={}
    for m in ["autovla","diffusiondrive","diffusiondrivev2","transfuser"]:
        p=f"{CMP}/{m}.json"
        if os.path.exists(p): MODELS[m]=json.load(open(p))
print("models:",list(MODELS))

def agents_in_t0(scene):
    """Return list over future frames k of (xy[N,2], vel[N,2], is_vru[N]) in the t=0 ego frame."""
    nh=scene.scene_metadata.num_history_frames
    fut=scene.get_future_trajectory()           # poses relative to current (t=0)
    poses=np.asarray(fut.poses)                  # [K,3] x,y,theta
    out=[]
    for k in range(len(poses)):
        fidx=nh+k
        if fidx>=len(scene.frames): break
        ann=scene.frames[fidx].annotations
        boxes=np.asarray(ann.boxes,dtype=float)
        if boxes.size==0:
            out.append((np.zeros((0,2)),np.zeros((0,2)),np.zeros((0,),bool))); continue
        loc=boxes[:,:2]                          # agent xy in frame-k ego coords
        x,y,th=poses[k]; c,s=math.cos(th),math.sin(th)
        R=np.array([[c,-s],[s,c]])
        xy_t0=(loc@R.T)+np.array([x,y])          # into t=0 frame
        vel=np.asarray(ann.velocity_3d,dtype=float)[:,:2] if np.asarray(ann.velocity_3d).size else np.zeros((len(loc),2))
        names=ann.names
        is_vru=np.array([any(kw in (nm or "").lower() for kw in VRU_KEYS) for nm in names])
        out.append((xy_t0,vel,is_vru))
    return out

def ego_series(xy):
    p=np.array([[0.0,0.0]]+list(xy),dtype=float)
    v=np.diff(p,axis=0)/DT
    sp=np.hypot(v[:,0],v[:,1])
    return p[1:], sp                              # positions (per future frame), speed

def interact(ego_xy, agents):
    pos,sp=ego_series(ego_xy)
    K=min(len(pos),len(agents))
    thw=ttc=lead_gap=math.inf; dmin=dvru=math.inf
    for k in range(K):
        exy=pos[k]; espeed=max(sp[k],1e-3)
        axy,avel,vru=agents[k]
        if len(axy)==0: continue
        d=np.hypot(axy[:,0]-exy[0],axy[:,1]-exy[1])
        dmin=min(dmin,float(d.min()))
        if vru.any(): dvru=min(dvru,float(d[vru].min()))
        # lead = ahead (x>ego_x) within lateral corridor
        ahead=(axy[:,0]>exy[0])&(np.abs(axy[:,1]-exy[1])<CORRIDOR)
        if ahead.any():
            gap=axy[ahead,0]-exy[0]
            j=int(np.argmin(gap)); g=float(gap[j])
            lead_gap=min(lead_gap,g)
            thw=min(thw,g/espeed)
            close=espeed-float(avel[ahead][j,0])
            if close>0.1: ttc=min(ttc,g/close)
    f=lambda v: (None if math.isinf(v) else round(v,3))
    return {"thw_min":f(thw),"ttc_min":f(ttc),"lead_gap_min":f(lead_gap),
            "dist_any_min":f(dmin),"dist_vru_min":f(dvru)}

def main():
    toks=[t for t in MODELS["autovla"] if t!="_meta"] if "autovla" in MODELS else list(ST)
    sf=SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,
                   has_route=True,max_scenes=None,log_names=None,tokens=toks)
    loader=SceneLoader(data_path=Path(f"{DATA}/navsim_logs/test"),
                       sensor_blobs_path=Path(f"{DATA}/sensor_blobs/test"),
                       scene_filter=sf,sensor_config=SensorConfig.build_no_sensors())
    common=[t for t in loader.tokens if t in ST][:CAP]
    print("scenes to process:",len(common))
    res={m:{} for m in MODELS}; res["human"]={}
    for i,tok in enumerate(common):
        if i%500==0: print(f"  {i}/{len(common)}",flush=True)
        try:
            sc=loader.get_scene_from_token(tok)
            ag=agents_in_t0(sc)
            res["human"][tok]=interact(np.asarray(sc.get_future_trajectory().poses)[:,:2],ag)
            for m,d in MODELS.items():
                if tok in d: res[m][tok]=interact(np.asarray(d[tok]),ag)
        except Exception as e:
            print("skip",tok,repr(e)[:90])
    json.dump(res,open(OUT,"w"))
    # summary: median over scenes where a lead/VRU exists
    print("\n=== environment-interaction (median over valid scenes) ===")
    print(f"{'who':<16}{'thw_min':>9}{'ttc_min':>9}{'lead_gap':>9}{'dist_any':>9}{'dist_vru':>9}")
    for who in ["human"]+list(MODELS):
        rows=res[who].values()
        def med(k):
            v=[r[k] for r in rows if r.get(k) is not None]
            return np.median(v) if v else float("nan")
        print(f"{who:<16}{med('thw_min'):>9.2f}{med('ttc_min'):>9.2f}{med('lead_gap_min'):>9.2f}"
              f"{med('dist_any_min'):>9.2f}{med('dist_vru_min'):>9.2f}")
    print("written",OUT)

if __name__=="__main__":
    main()
