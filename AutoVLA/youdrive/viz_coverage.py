"""Illustrate coverage = convex-hull area of a model's (F1,F2) points across scenes."""
import json, math, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
from scipy.spatial import ConvexHull
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
R=json.load(open("youdrive/style_metrics_full4.json"))
def pts(per): return np.array([(v["F1_assert"],v["F2_pace"]) for v in per.values()
                               if v["F1_assert"]==v["F1_assert"] and v["F2_pace"]==v["F2_pace"]])
panels=[("HUMAN-GT",R["human_per_token"],"black")]+[(m,R["per_token"][m],c) for m,c in
        [("autovla","#e6194B"),("diffusiondrive","#4363d8"),("diffusiondrivev2","#3cb44b"),("transfuser","#f58231")]]
fig,axes=plt.subplots(1,5,figsize=(22,5))
for ax,(name,per,c) in zip(axes,panels):
    P=pts(per)
    ax.scatter(P[:,0],P[:,1],s=3,alpha=0.15,c=c)
    h=ConvexHull(P); area=h.volume
    poly=P[h.vertices]; poly=np.vstack([poly,poly[0]])
    ax.plot(poly[:,0],poly[:,1],c=c,lw=2)
    ax.fill(poly[:,0],poly[:,1],c=c,alpha=0.12)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("F1 激进度"); ax.set_ylabel("F2 步调")
    ax.set_title(f"{name}\ncoverage(凸包面积)={area:.3f}",fontsize=11,fontweight="bold")
fig.suptitle("coverage = 每个场景一个点(F1,F2),所有点的凸包面积 = 该模型跨场景张开的风格范围 (0~1)",
             fontsize=13,fontweight="bold",y=1.04)
plt.tight_layout(); out="youdrive/metrics/09_coverage_hull.png"; plt.savefig(out,dpi=115,bbox_inches="tight"); print("saved",out)
