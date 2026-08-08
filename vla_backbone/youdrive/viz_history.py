"""History<->Future style coupling scatter (one feature per subplot)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False

R=json.load(open("youdrive/style_history_out.json"))
A=R["arrays"]; FEATS=list(A); r=R["pearson"]; n=R["n"]
LAB={"v_avg":"速度 v_avg (m/s)","peak_decel":"刹车 peak_decel (m/s²)","jerk_rms":"jerk_rms (m/s³)","lat_vmax":"横向 lat_vmax (m/s)"}
fig,axes=plt.subplots(2,2,figsize=(13,11))
for ax,k in zip(axes.ravel(),FEATS):
    h,f=np.array(A[k][0]),np.array(A[k][1])
    strong=r[k]>=0.5
    ax.scatter(h,f,s=10,alpha=0.3,c="#15803d" if strong else "#dc2626")
    lim=[min(h.min(),f.min()),max(h.max(),f.max())]
    ax.plot(lim,lim,"k--",lw=1,alpha=0.5)
    if h.std()>0:
        m,b=np.polyfit(h,f,1); xs=np.array(lim); ax.plot(xs,m*xs+b,c="#1d4ed8",lw=2)
    ax.set_xlabel(f"历史窗口 {LAB[k]}"); ax.set_ylabel(f"未来窗口 {LAB[k]}")
    ax.set_title(f"{k}   r={r[k]:+.3f}  {'✓ 可联动' if strong else '✗ 瞬态/不稳定'}",
                 fontsize=12,fontweight="bold",color="#15803d" if strong else "#dc2626")
fig.suptitle(f"驾驶历史 ↔ 未来 风格联动 (n={n} 场景, test split)\n"
             f"速度/刹车强相关→历史可作风格先验;jerk/横向弱→瞬态、场景驱动",
             fontsize=14,fontweight="bold",y=0.99)
plt.tight_layout(rect=[0,0,1,0.96])
out="youdrive/metrics/07_history_coupling.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
