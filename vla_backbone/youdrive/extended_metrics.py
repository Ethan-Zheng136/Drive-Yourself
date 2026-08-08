"""Broad numeric comparison: all currently-computable style dimensions (raw medians, 4049 scenes).
Kinematic dims from trajectories (human styletest vx/vy + model dumps) + env dims from env_interact_out.
"""
import json, math, os
import numpy as np
CMP="/mnt/pfs/zhengguantian/autovla/compare_full"
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ENV=json.load(open("youdrive/env_interact_out.json"))
DT=0.5

def feats_from_vxvy(vx,vy):
    vx=np.asarray(vx,float); vy=np.asarray(vy,float)
    sp=np.hypot(vx,vy)
    ax=np.diff(sp)/DT if len(sp)>1 else np.array([0.])
    ljerk=np.diff(ax)/DT if len(ax)>1 else np.array([0.])
    ay=np.diff(vy)/DT if len(vy)>1 else np.array([0.])
    ajerk=np.diff(ay)/DT if len(ay)>1 else np.array([0.])
    head=np.arctan2(vy,np.maximum(np.abs(vx),1e-3)*np.sign(vx+1e-9))
    yr=np.diff(head)/DT if len(head)>1 else np.array([0.])
    return {
        "v_avg":float(sp.mean()), "v_std":float(sp.std()),
        "peak_accel":float(ax.max()) if ax.size else 0., "peak_decel":float(-ax.min()) if ax.size else 0.,
        "long_jerk_rms":float(np.sqrt((ljerk**2).mean())) if ljerk.size else 0.,
        "lat_amax":float(np.abs(ay).max()) if ay.size else 0.,
        "lat_jerk_rms":float(np.sqrt((ajerk**2).mean())) if ajerk.size else 0.,
        "yaw_rate_max":float(np.abs(yr).max()) if yr.size else 0.,
    }

def model_vxvy(xy):
    p=np.array([[0.,0.]]+list(xy),float); d=np.diff(p,axis=0)/DT
    return d[:,0],d[:,1]

actors={}
# human from styletest
H=[]
for t,rec in ST.items():
    if rec.get("vx_ego") and rec.get("vy_ego"):
        H.append(feats_from_vxvy(rec["vx_ego"],rec["vy_ego"]))
actors["human"]=H
for m in ["autovla","diffusiondrive","diffusiondrivev2","transfuser"]:
    p=f"{CMP}/{m}.json"
    if not os.path.exists(p): continue
    d=json.load(open(p)); rows=[]
    for t,xy in d.items():
        if t=="_meta": continue
        vx,vy=model_vxvy(xy); rows.append(feats_from_vxvy(vx,vy))
    actors[m]=rows

KIN=["v_avg","v_std","peak_accel","peak_decel","long_jerk_rms","lat_amax","lat_jerk_rms","yaw_rate_max"]
ENVK=["thw_min","ttc_min","lead_gap_min","dist_any_min","dist_vru_min"]
def med(rows,k): 
    v=[r[k] for r in rows if r.get(k) is not None and not math.isnan(r[k])]
    return float(np.median(v)) if v else float("nan")
def envmed(who,k):
    v=[r[k] for r in ENV.get(who,{}).values() if r.get(k) is not None]
    return float(np.median(v)) if v else float("nan")

names=["human","autovla","diffusiondrive","diffusiondrivev2","transfuser"]
LABEL={"v_avg":"平均速度(m/s)","v_std":"速度波动(m/s)","peak_accel":"峰值加速(m/s2)","peak_decel":"峰值刹车(m/s2)",
 "long_jerk_rms":"纵向jerk(m/s3)","lat_amax":"横向加速度(m/s2)","lat_jerk_rms":"横向jerk(m/s3)","yaw_rate_max":"最大偏航率(rad/s)",
 "thw_min":"THW车头时距(s)","ttc_min":"min-TTC(s)","lead_gap_min":"前车间距(m)","dist_any_min":"最近他车(m)","dist_vru_min":"最近VRU(m)"}

print(f"{'维度':<22}" + "".join(f"{n[:9]:>11}" for n in names))
print("-"*(22+11*5))
print("【组A/B 自车动力学(轨迹直算)】")
for k in KIN:
    print(f"{LABEL[k]:<22}" + "".join(f"{med(actors[n],k):>11.3f}" for n in names))
print("【组C/D 环境交互(vs真实他车/行人)】")
for k in ENVK:
    print(f"{LABEL[k]:<22}" + "".join(f"{envmed(n,k):>11.2f}" for n in names))

out={"kin":{n:{k:med(actors[n],k) for k in KIN} for n in names},
     "env":{n:{k:envmed(n,k) for k in ENVK} for n in names},"labels":LABEL}
json.dump(out,open("youdrive/extended_metrics_out.json","w"),indent=2,ensure_ascii=False)
print("\nwritten youdrive/extended_metrics_out.json")
