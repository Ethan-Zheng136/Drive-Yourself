"""YDSP radar: DAI/PAI/SAI per model (percentile vs human)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
Y=json.load(open("youdrive/ydsp_out.json"))
axes=["DAI\n动力学激进","PAI\n步调激进","SAI\n社交激进"]
ang=np.linspace(0,2*np.pi,3,endpoint=False).tolist(); ang+=ang[:1]
COL={"human":"black","autovla":"#e6194B","diffusiondrive":"#4363d8","diffusiondrivev2":"#3cb44b",
     "transfuser":"#f58231","gtrs":"#911eb4","hydra_mdp":"#42d4f4","wote":"#a9a9a9","goalflow":"#f032e6"}
fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
for name in ["human","autovla","diffusiondrive","diffusiondrivev2","transfuser","gtrs","hydra_mdp","wote","goalflow"]:
    v=[Y[name]["DAI"],Y[name]["PAI"],Y[name]["SAI"]]; v+=v[:1]
    lw=3 if name=="human" else 1.8; ls="--" if name=="human" else "-"
    ax.plot(ang,v,ls,lw=lw,color=COL[name],label=name)
    ax.fill(ang,v,color=COL[name],alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes,fontsize=11)
ax.set_ylim(0,100); ax.set_yticks([25,50,75,100]); ax.set_yticklabels(["25","50\n人类","75","100"],fontsize=8)
ax.set_title("YDSP 风格画像:DAI/PAI/SAI(相对人类百分位)\n虚线黑=人类(50);离中心越远越激进",fontsize=13,fontweight="bold",pad=20)
ax.legend(loc="upper right",bbox_to_anchor=(1.25,1.1),fontsize=10)
out="youdrive/metrics/14_ydsp_radar_7models.png"; plt.tight_layout(); plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
