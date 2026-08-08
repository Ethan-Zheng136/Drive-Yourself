"""Environment-interaction metrics: one metric per panel (median over valid scenes)."""
import json, math, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
R=json.load(open("youdrive/env_interact_out.json"))
actors=["human","autovla","diffusiondrive","diffusiondrivev2","transfuser"]
COL={"human":"black","autovla":"#e6194B","diffusiondrive":"#4363d8","diffusiondrivev2":"#3cb44b","transfuser":"#f58231"}
def med(who,k):
    v=[r[k] for r in R[who].values() if r.get(k) is not None]
    return float(np.median(v)) if v else float("nan")
PANELS=[("thw_min","THW 车头时距 (s)","小 = 跟车凶"),
        ("ttc_min","min-TTC (s)","小 = 跟车凶"),
        ("lead_gap_min","前车纵向间距 (m)","小 = 贴得近"),
        ("dist_any_min","对任意他车最近距离 (m)","小 = 凑得近"),
        ("dist_vru_min","对行人/骑行者最近距离 (m)","小 = 离VRU近")]
fig,axes=plt.subplots(2,3,figsize=(17,9)); axes=axes.ravel()
for ax,(k,title,hint) in zip(axes,PANELS):
    vals=[med(a,k) for a in actors]
    ax.bar(actors,vals,color=[COL[a] for a in actors],edgecolor="k",linewidth=0.4)
    ax.axhline(vals[0],color="black",ls=":",lw=1)   # human reference
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*0.01,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_title(f"{title}\n({hint};虚线=人类)",fontsize=11,fontweight="bold")
    ax.set_xticklabels(actors,rotation=20,ha="right",fontsize=8)
axes[-1].axis("off")
axes[-1].text(0.05,0.7,"环境交互维度(组 B)关键结论:\n\n"
    "• AutoVLA 五项几乎完全贴合人类\n  → VLM 学到了类人的社交裕度\n\n"
    "• DiffusionDriveV2 THW/TTC 最小\n  → 跟车最凶(和它高 F1 一致)\n\n"
    "• DD / TransFuser 裕度更大\n  → 更保守\n\n"
    "纯轨迹运动学看不出这些,必须和环境联动",fontsize=11,va="top")
fig.suptitle("环境交互风格(组 B):模型预测 ego 轨迹 vs 场景真实他车/行人 (4049 场景中位数)",
             fontsize=13,fontweight="bold",y=1.01)
plt.tight_layout(); out="youdrive/metrics/10_env_interaction.png"; plt.savefig(out,dpi=115,bbox_inches="tight"); print("saved",out)
