"""YDSP (paired / scene-controlled): for each token, deviation of model from HUMAN on the SAME scene.
Removes scenario confound (each scene is its own control). Output = median per-token deviation in
human-sigma units, oriented so + = more aggressive than the human did here. Human baseline = 0.
DAI/PAI from kinematic paired diffs (all tokens); SAI from env paired diffs (tokens with a lead/agent).
"""
import json, math, numpy as np
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ENV=json.load(open("youdrive/env_interact_out.json"))
LOAD=json.load(open("youdrive/pca_style_out.json"))["loadings"]
CMP="/mnt/pfs/zhengguantian/autovla/compare_full"; DT=0.5
KIN=["v_avg","v_std","peak_acc","peak_dec","long_jerk","lat_amax","lat_jerk"]
ENVK=["thw_min","ttc_min","dist_any","dist_vru"]
DUMP={"gtrs":f"{CMP}/gtrs_r.json","hydra_mdp":f"{CMP}/hydra_mdp_r.json"}
MODELS=["autovla","diffusiondrive","diffusiondrivev2","transfuser","gtrs","hydra_mdp","wote","goalflow"]

def kin(vx,vy):
    vx=np.asarray(vx,float);vy=np.asarray(vy,float);sp=np.hypot(vx,vy)
    ax=np.diff(sp)/DT if len(sp)>1 else np.array([0.]); lj=np.diff(ax)/DT if len(ax)>1 else np.array([0.])
    ay=np.diff(vy)/DT if len(vy)>1 else np.array([0.]); aj=np.diff(ay)/DT if len(ay)>1 else np.array([0.])
    return {"v_avg":float(sp.mean()),"v_std":float(sp.std()),"peak_acc":float(ax.max()) if ax.size else 0,
            "peak_dec":float(-ax.min()) if ax.size else 0,"long_jerk":float(np.sqrt((lj**2).mean())) if lj.size else 0,
            "lat_amax":float(np.abs(ay).max()) if ay.size else 0,"lat_jerk":float(np.sqrt((aj**2).mean())) if aj.size else 0}
def mvxvy(xy): p=np.array([[0.,0.]]+list(xy),float);d=np.diff(p,axis=0)/DT;return d[:,0],d[:,1]

# human per-token kin + global sigma per feature
hkin={t:kin(r["vx_ego"],r["vy_ego"]) for t,r in ST.items() if r.get("vx_ego") and r.get("vy_ego")}
allk={k:np.array([v[k] for v in hkin.values()]) for k in KIN}
sigK={k:(allk[k].std() or 1.0) for k in KIN}
henv=ENV["human"]
allenv={k:np.array([r[k] for r in henv.values() if r.get(k) is not None]) for k in ENVK}
sigE={k:(allenv[k].std() or 1.0) for k in ENVK}
L={f:np.array(LOAD[f]) for f in KIN+ENVK}   # [3] per feature
# orient: DAI=-PC1, PAI=+PC2, SAI=-PC3
ORI=np.array([-1,1,-1])

def model_kin(name):
    d=json.load(open(DUMP.get(name,f"{CMP}/{name}.json")))
    out={}
    for t,xy in d.items():
        if t=="_meta": continue
        out[t]=kin(*mvxvy(xy))
    return out

print(f"{'model':<16}{'DAI':>8}{'PAI':>8}{'SAI':>8}   (per-scene 偏离人类, σ单位; 0=和人类一样, +=更激进)")
print("-"*52)
out={}
for m in MODELS:
    mk=model_kin(m); me=ENV.get(m,{})
    # DAI(PC1), PAI(PC2) from kinematic paired diffs
    dai=[]; pai=[]
    for t in mk:
        if t not in hkin: continue
        z={k:(mk[t][k]-hkin[t][k])/sigK[k] for k in KIN}
        v1=sum(z[k]*L[k][0] for k in KIN); v2=sum(z[k]*L[k][1] for k in KIN)
        dai.append(v1*ORI[0]); pai.append(v2*ORI[1])
    # SAI(PC3) from env paired diffs (tokens with both)
    sai=[]
    for t in me:
        h=henv.get(t)
        if not h: continue
        zs=[]; 
        ok=True
        for k in ENVK:
            if me[t].get(k) is None or h.get(k) is None: continue
            zs.append(((me[t][k]-h[k])/sigE[k])*L[k][2])
        if zs: sai.append(sum(zs)*ORI[2])
    DAI=float(np.median(dai)); PAI=float(np.median(pai)); SAI=float(np.median(sai)) if sai else float("nan")
    out[m]={"DAI":round(DAI,3),"PAI":round(PAI,3),"SAI":round(SAI,3),"n_kin":len(dai),"n_env":len(sai)}
    print(f"{m:<16}{DAI:>8.2f}{PAI:>8.2f}{SAI:>8.2f}")
json.dump(out,open("youdrive/ydsp_paired_out.json","w"),indent=2)
print("\nwritten youdrive/ydsp_paired_out.json  (0=human baseline; sigma units)")
