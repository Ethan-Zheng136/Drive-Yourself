"""YDSP radar with PER-AXIS independent scaling (each axis spans its own min..max)
so small spreads on PAI/SAI become visible. Axis labels show the real value range."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
Y=json.load(open("youdrive/ydsp_out.json"))
order=["human","autovla","diffusiondrive","diffusiondrivev2","transfuser","gtrs","hydra_mdp","wote","goalflow"]
order=[m for m in order if m in Y]
AX=["DAI","PAI","SAI"]
COL={"human":"black","autovla":"#e6194B","diffusiondrive":"#4363d8","diffusiondrivev2":"#3cb44b",
     "transfuser":"#f58231","gtrs":"#911eb4","hydra_mdp":"#42d4f4","wote":"#a9a9a9","goalflow":"#f032e6"}
# per-axis min/max over all actors (+small padding)
vals={a:np.array([Y[m][a] for m in order]) for a in AX}
lo={a:vals[a].min() for a in AX}; hi={a:vals[a].max() for a in AX}
pad={a:max((hi[a]-lo[a])*0.08,0.5) for a in AX}
lo={a:lo[a]-pad[a] for a in AX}; hi={a:hi[a]+pad[a] for a in AX}
def norm(a,v): return (v-lo[a])/(hi[a]-lo[a])
ang=np.linspace(0,2*np.pi,3,endpoint=False).tolist(); ang+=ang[:1]
fig,ax=plt.subplots(figsize=(8.5,8.5),subplot_kw=dict(polar=True))
for m in order:
    r=[norm(a,Y[m][a]) for a in AX]; r+=r[:1]
    lw=3 if m=="human" else 1.8; ls="--" if m=="human" else "-"
    ax.plot(ang,r,ls,lw=lw,color=COL[m],label=m); ax.fill(ang,r,color=COL[m],alpha=0.05)
ax.set_ylim(0,1); ax.set_yticklabels([])
labs=[f"{a}\n[{lo[a]+pad[a]-pad[a]:.0f}…]" for a in AX]  # placeholder
labs=[f"{AX[i]}\n{vals[AX[i]].min():.0f}→{vals[AX[i]].max():.0f}" for i in range(3)]
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs,fontsize=12,fontweight="bold")
# mark human(=50 each) ring position per axis as small ticks
for i,a in enumerate(AX):
    for vv in [vals[a].min(),vals[a].max()]:
        ax.text(ang[i], norm(a,vv), f"{vv:.0f}", fontsize=7, color="#666", ha="center")
ax.set_title("YDSP 风格画像(每轴独立缩放,凸显差异)\n注:各轴量程不同(标在轴名下),视觉幅度不可跨轴比较",
             fontsize=12,fontweight="bold",pad=22)
ax.legend(loc="upper right",bbox_to_anchor=(1.28,1.12),fontsize=9)
out="youdrive/metrics/15_ydsp_radar_scaled.png"; plt.tight_layout(); plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
