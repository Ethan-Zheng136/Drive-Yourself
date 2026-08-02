"""Raw-physical dose-response: THW / min-TTC / lead_gap vs alpha, with DDv2(teacher) & human refs."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name(); plt.rcParams["axes.unicode_minus"]=False
ENV=json.load(open("youdrive/env_interact_out.json"))
def med(who,k):
    v=[r[k] for r in ENV[who].values() if r.get(k) is not None]; return float(np.median(v)) if v else float("nan")
# alpha points
A=[0.0,0.5,1.0]; KEY={0.0:"autovla",0.5:"persona_a0.5",1.0:"persona_a1.0"}
panels=[("thw_min","THW 车头时距 (s)"),("ttc_min","min-TTC (s)"),("lead_gap_min","前车间距 (m)")]
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,(k,title) in zip(axes,panels):
    ys=[med(KEY[a],k) for a in A]
    ax.plot(A,ys,'-o',color="#e6194B",lw=2.6,ms=9,label="base+α·v_DDv2")
    dd=med("diffusiondrivev2",k)
    ax.scatter([1.25],[dd],marker="*",s=420,color="#3cb44b",edgecolors="black",zorder=10,label="DDv2 实际(老师)")
    ax.annotate(f"{dd:.2f}",(1.25,dd),textcoords="offset points",xytext=(8,0),fontsize=10,color="#3cb44b",fontweight="bold")
    ax.axhline(dd,color="#3cb44b",ls="--",lw=1.2,alpha=0.6)
    ax.axhline(med("human",k),color="black",ls=":",lw=2,label="人类")
    for a,y in zip(A,ys): ax.annotate(f"{y:.2f}",(a,y),textcoords="offset points",xytext=(6,6),fontsize=9)
    ax.set_xlabel("α (persona 强度)"); ax.set_ylabel(title); ax.set_title(title,fontsize=12,fontweight="bold")
    ax.set_xticks([0,0.5,1.0,1.25]); ax.set_xticklabels(["0","0.5","1.0","DDv2"])
    ax.invert_yaxis()  # 小=更激进,放上面更直观
axes[0].legend(fontsize=9,loc="best")
fig.suptitle("DDv2-persona dose-response(原始物理量):α↑ → 跟车更紧/更凶,α=1 逼近 DDv2 老师\n"
             "(y轴已翻转,越往上越激进)",fontsize=13,fontweight="bold",y=1.02)
plt.tight_layout(); out="youdrive/metrics/17_dose_raw_env.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
