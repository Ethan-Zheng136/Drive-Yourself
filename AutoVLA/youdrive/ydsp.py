"""YDSP = YouDrive Driving Style Profile = {DAI, PAI, SAI}.
DAI Dynamic Aggressiveness Index  (PC1, jerk/accel; oriented so high=harsh)
PAI Pace    Aggressiveness Index  (PC2, speed vs braking; high=brisk)
SAI Social  Aggressiveness Index  (PC3, THW/TTC/dist; high=tailgating)
Reported as percentile vs the human factor distribution (human=50, >50 more aggressive than typical human).
"""
import json, math
import numpy as np
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ENV=json.load(open("youdrive/env_interact_out.json"))
LOAD=json.load(open("youdrive/pca_style_out.json"))["loadings"]
CMP="/mnt/pfs/zhengguantian/autovla/compare_full"; DT=0.5
ALL=["v_avg","v_std","peak_acc","peak_dec","long_jerk","lat_amax","lat_jerk","thw_min","ttc_min","dist_any","dist_vru"]
L=np.array([LOAD[f] for f in ALL]).T          # [3,11]
ORIENT=np.array([-1,+1,-1])                    # DAI=-PC1, PAI=+PC2, SAI=-PC3 (higher=aggressive)

def kin(vx,vy):
    vx=np.asarray(vx,float); vy=np.asarray(vy,float); sp=np.hypot(vx,vy)
    ax=np.diff(sp)/DT if len(sp)>1 else np.array([0.])
    lj=np.diff(ax)/DT if len(ax)>1 else np.array([0.])
    ay=np.diff(vy)/DT if len(vy)>1 else np.array([0.])
    aj=np.diff(ay)/DT if len(ay)>1 else np.array([0.])
    return [float(sp.mean()),float(sp.std()),float(ax.max()) if ax.size else 0,
            float(-ax.min()) if ax.size else 0,float(np.sqrt((lj**2).mean())) if lj.size else 0,
            float(np.abs(ay).max()) if ay.size else 0,float(np.sqrt((aj**2).mean())) if aj.size else 0]
def mdl_vxvy(xy):
    p=np.array([[0.,0.]]+list(xy),float); d=np.diff(p,axis=0)/DT; return d[:,0],d[:,1]

def row(kin7,env4):
    if any(v is None for v in env4): return None
    return np.array(kin7+list(env4),float)

# human rows -> standardization params + human factor distribution
H=[]
for t,rec in ST.items():
    if not(rec.get("vx_ego") and rec.get("vy_ego")): continue
    e=ENV["human"].get(t,{}); ev=[e.get("thw_min"),e.get("ttc_min"),e.get("dist_any_min"),e.get("dist_vru_min")]
    r=row(kin(rec["vx_ego"],rec["vy_ego"]),ev)
    if r is not None: H.append(r)
H=np.array(H)
# ROBUST: percentile-normalize each feature vs human (bounded 0-1), center at 0.5.
HSORT=[np.sort(H[:,j]) for j in range(H.shape[1])]
def to_pct(X):                                  # X[n,11] -> centered percentiles in [-0.5,0.5]
    P=np.zeros_like(X,dtype=float)
    for j in range(X.shape[1]):
        P[:,j]=np.searchsorted(HSORT[j],X[:,j],side="right")/len(HSORT[j])
    return P-0.5
def factors(X):                                # robust factor scores (outlier-capped)
    return (to_pct(X)@L.T)*ORIENT
HF=factors(H)                                   # human factor distribution
def pct(val,dist): return float((dist<val).mean()*100)

DUMP_PATH={"gtrs":f"{CMP}/gtrs_r.json","hydra_mdp":f"{CMP}/hydra_mdp_r.json"}  # resampled to 8pts/0.5s
def model_rows(name):
    d=json.load(open(DUMP_PATH.get(name,f"{CMP}/{name}.json"))); rows=[]
    for t,xy in d.items():
        if t=="_meta": continue
        e=ENV[name].get(t,{}); ev=[e.get("thw_min"),e.get("ttc_min"),e.get("dist_any_min"),e.get("dist_vru_min")]
        r=row(kin(*mdl_vxvy(xy)),ev)
        if r is not None: rows.append(r)
    return np.array(rows)

print(f"{'model':<16}{'DAI':>7}{'PAI':>7}{'SAI':>7}   (percentile vs human, 50=human median, >50 更激进)")
print("-"*52)
out={"_def":"percentile vs human; 50=human median; higher=more aggressive"}
# human baseline = median percentile of its own dist = 50 each
print(f"{'HUMAN':<16}{50.0:>7.0f}{50.0:>7.0f}{50.0:>7.0f}")
out["human"]={"DAI":50.0,"PAI":50.0,"SAI":50.0}
for m in ["autovla","diffusiondrive","diffusiondrivev2","transfuser","gtrs","hydra_mdp","wote","goalflow"]:
    F=factors(model_rows(m)); med=np.median(F,axis=0)
    dai,pai,sai=[pct(med[k],HF[:,k]) for k in range(3)]
    print(f"{m:<16}{dai:>7.0f}{pai:>7.0f}{sai:>7.0f}")
    out[m]={"DAI":round(dai,1),"PAI":round(pai,1),"SAI":round(sai,1)}
json.dump(out,open("youdrive/ydsp_out.json","w"),indent=2)
print("\nwritten youdrive/ydsp_out.json")
