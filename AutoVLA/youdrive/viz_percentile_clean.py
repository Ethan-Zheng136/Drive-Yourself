"""Clean single-feature percentile explainer: percentile = shaded area to the left of the model's value."""
import json, math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
sys.path.insert(0, os.path.dirname(__file__))
from style_metrics import human_features
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False

ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
R=json.load(open("youdrive/style_metrics_anc.json"))
# use a representative token + AutoVLA, scenario = its scenario_type
tok=sorted(R["human_per_token"])[0]; ctx=ST[tok]["scenario_type"]
av=R["per_token"]["AutoVLA"][tok]["raw"]

# human population for this scenario_type
pop={}
for t,rec in ST.items():
    if rec.get("scenario_type")!=ctx: continue
    f=human_features(rec)
    if f:
        for k,v in f.items(): pop.setdefault(k,[]).append(v)

panels=[("v_avg","速度 v_avg (m/s)","快","F2 步调"),
        ("peak_decel","最猛刹车 peak_decel (m/s²)","刹得猛","F1 激进度")]
fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,(k,lab,more,fac) in zip(axes,panels):
    arr=np.array([x for x in pop[k] if not (isinstance(x,float) and math.isnan(x))],float)
    val=av[k]; pct=(arr<val).mean()*100
    cnt,bins,_=ax.hist(arr,bins=40,color="#cbd5e1",edgecolor="#94a3b8")
    # shade area to the LEFT of model value = percentile
    left=arr[arr<val]
    ax.hist(left,bins=bins,color="#86efac",edgecolor="#94a3b8")
    ax.axvline(val,color="#dc2626",lw=3)
    ax.text(val,cnt.max()*1.02,f"AutoVLA\n{val:.2f}",color="#dc2626",ha="center",fontsize=11,fontweight="bold")
    ax.set_xlabel(lab,fontsize=12); ax.set_ylabel(f"人类场景数 (scenario={ctx})",fontsize=11)
    ax.set_title(f"{fac}  的一个分量:{k}\nAutoVLA 在第 {pct:.0f} 百分位 = 比 {pct:.0f}% 的人{more}",
                 fontsize=13,fontweight="bold")
    ax.text(0.98,0.7,f"绿色面积\n= 左侧人占比\n= {pct:.0f}%",transform=ax.transAxes,ha="right",
            fontsize=11,color="#15803d",bbox=dict(boxstyle="round",fc="white",ec="#15803d"))
fig.suptitle("直方图联动怎么读:把模型的原始值(红线)钉在该场景全部人类的分布上,\n"
             "红线左边的人占比(绿色面积)就是这个特征的百分位;F1/F2 是若干这种百分位的平均",
             fontsize=13,fontweight="bold",y=1.02)
plt.tight_layout()
out="youdrive/metrics/06b_percentile_clean.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out,"pct done")
