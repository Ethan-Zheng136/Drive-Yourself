"""DDv2-persona dose-response: DAI/PAI/SAI vs alpha (0/0.5/1.0). Robust percentile-vs-human."""
import json, math, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name(); plt.rcParams["axes.unicode_minus"]=False
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ENV=json.load(open("youdrive/env_interact_out.json"))
LOAD=json.load(open("youdrive/pca_style_out.json"))["loadings"]
DT=0.5
KIN=["v_avg","v_std","peak_acc","peak_dec","long_jerk","lat_amax","lat_jerk"]; ENVK=["thw_min","ttc_min","dist_any","dist_vru"]
ALL=KIN+ENVK; L=np.array([LOAD[f] for f in ALL]).T; ORI=np.array([-1,1,-1])
# alpha -> (dump_path, env_key)
PTS=[(0.0,"/mnt/pfs/zhengguantian/autovla/compare_full/autovla.json","autovla"),
     (0.5,"/mnt/pfs/zhengguantian/autovla/compare_persona/persona_a0.5.json","persona_a0.5"),
     (1.0,"/mnt/pfs/zhengguantian/autovla/compare_persona/persona_a1.0.json","persona_a1.0")]

def kin(vx,vy):
    vx=np.asarray(vx,float);vy=np.asarray(vy,float);sp=np.hypot(vx,vy)
    ax=np.diff(sp)/DT if len(sp)>1 else np.array([0.]);lj=np.diff(ax)/DT if len(ax)>1 else np.array([0.])
    ay=np.diff(vy)/DT if len(vy)>1 else np.array([0.]);aj=np.diff(ay)/DT if len(ay)>1 else np.array([0.])
    return {"v_avg":float(sp.mean()),"v_std":float(sp.std()),"peak_acc":float(ax.max()) if ax.size else 0,
            "peak_dec":float(-ax.min()) if ax.size else 0,"long_jerk":float(np.sqrt((lj**2).mean())) if lj.size else 0,
            "lat_amax":float(np.abs(ay).max()) if ay.size else 0,"lat_jerk":float(np.sqrt((aj**2).mean())) if aj.size else 0}
def hk(rec):
    f=kin(rec["vx_ego"],rec["vy_ego"]); 
    e=ENV["human"].get(rec["token"],{}); 
    for k in ENVK: f[k]=e.get(k)
    return f
def mvxvy(xy): p=np.array([[0.,0.]]+list(xy),float);d=np.diff(p,axis=0)/DT;return d[:,0],d[:,1]
# human reference sorted arrays
H=[]
for t,r in ST.items():
    if r.get("vx_ego"): H.append(hk(r))
HSORT={}
for j,k in enumerate(ALL):
    v=np.array([d[k] for d in H if d.get(k) is not None],float); HSORT[k]=np.sort(v)
def pct(val,arr): 
    return np.nan if val is None or (isinstance(val,float) and math.isnan(val)) else np.searchsorted(arr,val,side="right")/len(arr)
def factors_for(dump,envkey):
    d=json.load(open(dump)); rows=[]
    for t,xy in d.items():
        if t=="_meta": continue
        f=kin(*mvxvy(xy)); e=ENV[envkey].get(t,{})
        for k in ENVK: f[k]=e.get(k)
        z=np.array([ (pct(f[k],HSORT[k]) - 0.5) if (f[k] is not None and not(isinstance(f[k],float) and math.isnan(f[k]))) else 0.0 for k in ALL])
        rows.append((z@L.T)*ORI)
    R=np.array(rows); return np.median(R,axis=0)

res={}
for a,dump,ek in PTS:
    F=factors_for(dump,ek); res[a]={"DAI":float(F[0]),"PAI":float(F[1]),"SAI":float(F[2])}
    print(f"alpha={a}: DAI={F[0]:+.3f} PAI={F[1]:+.3f} SAI={F[2]:+.3f}  (centered, +=更激进 vs 人类中位)")
json.dump(res,open("youdrive/ydsp_dose_out.json","w"),indent=2)
# plot
al=[a for a,_,_ in PTS]
fig,ax=plt.subplots(figsize=(8,6))
for k,c in [("DAI","#e6194B"),("PAI","#4363d8"),("SAI","#3cb44b")]:
    ax.plot(al,[res[a][k] for a in al],'-o',color=c,lw=2.5,ms=8,label=k)
ax.axhline(0,color="#999",ls=":"); ax.set_xlabel("α (persona 强度: base + α·v_DDv2)",fontsize=12)
ax.set_ylabel("风格因子(相对人类中位, +=更激进)",fontsize=12)
ax.set_title("DDv2-persona dose-response: 风格随 α 的变化\n(单调上升 = persona vector 可控)",fontsize=13,fontweight="bold")
ax.legend(fontsize=11); ax.set_xticks(al)
plt.tight_layout(); out="youdrive/metrics/16_persona_dose_response.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
