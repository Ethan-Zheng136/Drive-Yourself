"""Data-driven composite factors: PCA on HUMAN per-scene style features.
Shows how many independent 'style factors' the raw metrics collapse into, and the loadings.
Human data only -> the factor basis is defined by real human style variation (no model jitter)."""
import json, math
import numpy as np
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ENV=json.load(open("youdrive/env_interact_out.json"))["human"]
DT=0.5

def kin(vx,vy):
    vx=np.asarray(vx,float); vy=np.asarray(vy,float); sp=np.hypot(vx,vy)
    ax=np.diff(sp)/DT if len(sp)>1 else np.array([0.])
    lj=np.diff(ax)/DT if len(ax)>1 else np.array([0.])
    ay=np.diff(vy)/DT if len(vy)>1 else np.array([0.])
    aj=np.diff(ay)/DT if len(ay)>1 else np.array([0.])
    return [float(sp.mean()),float(sp.std()),float(ax.max()) if ax.size else 0,
            float(-ax.min()) if ax.size else 0,float(np.sqrt((lj**2).mean())) if lj.size else 0,
            float(np.abs(ay).max()) if ay.size else 0,float(np.sqrt((aj**2).mean())) if aj.size else 0]
KCOLS=["v_avg","v_std","peak_acc","peak_dec","long_jerk","lat_amax","lat_jerk"]
ECOLS=["thw_min","ttc_min","dist_any","dist_vru"]
ALL=KCOLS+ECOLS

rows=[]
for t,rec in ST.items():
    if not(rec.get("vx_ego") and rec.get("vy_ego")): continue
    e=ENV.get(t,{})
    ev=[e.get("thw_min"),e.get("ttc_min"),e.get("dist_any_min"),e.get("dist_vru_min")]
    if any(v is None for v in ev): continue          # need complete env row for PCA
    rows.append(kin(rec["vx_ego"],rec["vy_ego"])+ev)
X=np.array(rows,float)
print(f"PCA on {X.shape[0]} human scenes x {X.shape[1]} features (complete rows)")
# standardize
mu=X.mean(0); sd=X.std(0)+1e-9; Z=(X-mu)/sd
# PCA via SVD
U,S,Vt=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
var=S**2; evr=var/var.sum()
print("\nexplained variance ratio (per PC):")
cum=0
for i,r in enumerate(evr[:6]):
    cum+=r; print(f"  PC{i+1}: {r*100:5.1f}%   cumulative {cum*100:5.1f}%")
print("\nloadings (how each raw metric weights into PC1..PC3):")
print(f"{'feature':<12}{'PC1':>8}{'PC2':>8}{'PC3':>8}")
for j,name in enumerate(ALL):
    print(f"{name:<12}{Vt[0,j]:>8.2f}{Vt[1,j]:>8.2f}{Vt[2,j]:>8.2f}")
import json as J
J.dump({"evr":evr[:6].tolist(),"loadings":{ALL[j]:[float(Vt[k,j]) for k in range(3)] for j in range(len(ALL))}},
       open("youdrive/pca_style_out.json","w"),indent=2)
print("\nwritten youdrive/pca_style_out.json")
