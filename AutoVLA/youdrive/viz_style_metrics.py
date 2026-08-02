"""Visualize style-metrics output: (a) scenario elasticity bar, (b) 2D style plane scatter."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False

R=json.load(open("youdrive/style_metrics_anc.json"))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17,7))

# (a) elasticity bar — recompute over ALL scenarios from a fuller run if present; use top set
el=R["top_elastic_scenarios"]
names=list(el); vals=[el[n]["elasticity"] for n in names]; ns=[el[n]["n"] for n in names]
colors=["#15803d" if v>=0.19 else ("#f59e0b" if v>=0.12 else "#dc2626") for v in vals]
y=np.arange(len(names))
ax1.barh(y,vals,color=colors)
ax1.set_yticks(y); ax1.set_yticklabels([f"{n}  (n={c})" for n,c in zip(names,ns)],fontsize=10)
ax1.invert_yaxis(); ax1.set_xlabel("风格弹性 elasticity (人类行为方差)",fontsize=11)
ax1.set_title("场景风格弹性:只在高弹性场景能测出风格\n红=低弹性(lane_following 占58%却最低)",fontsize=12,fontweight="bold")
for i,(v,c) in enumerate(zip(vals,ns)): ax1.text(v+0.004,i,f"{v:.3f}",va="center",fontsize=9)

# (b) 2D style plane
MC={"AutoVLA":"#e6194B","DiffusionDrive":"#4363d8","DiffusionDriveV2":"#3cb44b","TransFuser":"#f58231"}
for name,per in R["per_token"].items():
    xs=[v["F1_assert"] for v in per.values() if v["F1_assert"]==v["F1_assert"]]
    ys=[v["F2_pace"] for v in per.values() if v["F2_pace"]==v["F2_pace"]]
    ax2.scatter(xs,ys,c=MC.get(name,"#999"),label=name,s=80,alpha=0.8,edgecolors="k",linewidths=0.5)
hx=[v["F1_assert"] for v in R["human_per_token"].values() if v["F1_assert"]==v["F1_assert"]]
hy=[v["F2_pace"] for v in R["human_per_token"].values() if v["F2_pace"]==v["F2_pace"]]
ax2.scatter(hx,hy,c="black",marker="*",s=320,label="Human GT",edgecolors="white",linewidths=1,zorder=10)
ax2.axhline(0.5,color="#bbb",ls=":"); ax2.axvline(0.5,color="#bbb",ls=":")
ax2.set_xlim(0,1.02); ax2.set_ylim(0.4,1.02)
ax2.set_xlabel("F1  Assertiveness (裕度/减速/jerk 百分位)",fontsize=11)
ax2.set_ylabel("F2  Pace (速度/横向活动 百分位)",fontsize=11)
hm=R.get("cross_model_homogeneity",{}).get("mean_pairwise_style_dist",0)
ax2.set_title(f"2D 风格平面:模型挤成一团,远离人类\ncross-model homogeneity={hm:.3f} (0=完全相同)",fontsize=12,fontweight="bold")
ax2.legend(fontsize=10,loc="lower left")
plt.tight_layout()
out="youdrive/style_metrics.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
