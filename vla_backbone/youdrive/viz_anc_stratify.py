"""ANC-stratified F1 / F2 grouped bars (one metric per panel)."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
S=json.load(open("youdrive/anc_stratify_out.json"))
ORDER=["A","N","C"]; rows=["HUMAN-GT"]+list(S["models"])
def get(name,axis): 
    src=S["human"] if name=="HUMAN-GT" else S["models"][name]
    return [src[a][axis] for a in ORDER]
COL={"HUMAN-GT":"black","autovla":"#e6194B","diffusiondrive":"#4363d8","diffusiondrivev2":"#3cb44b","transfuser":"#f58231"}
fig,axes=plt.subplots(1,2,figsize=(16,6))
for ax,axis,title,sub in [(axes[0],"F1","指标 F1 激进度 按 ANC 分层","纵向激进度(减速/jerk 百分位)"),
                          (axes[1],"F2","指标 F2 步调 按 ANC 分层","速度/横向 百分位")]:
    x=np.arange(3); w=0.16
    for i,name in enumerate(rows):
        ax.bar(x+i*w,get(name,axis),w,label=name,color=COL[name],edgecolor="k",linewidth=0.4)
    ax.set_xticks(x+2*w); ax.set_xticklabels(["A 激进\n(n=538)","N 中性\n(n=3275)","C 保守\n(n=236)"])
    ax.set_ylabel(f"{axis} 百分位"); ax.set_ylim(0,1); ax.axhline(0.5,color="#bbb",ls=":")
    ax.set_title(f"{title}\n{sub}",fontsize=12,fontweight="bold"); ax.legend(fontsize=8,ncol=2)
fig.suptitle("人类标注 A/N/C 场景上,各模型与人类的风格位\n"
             "关键:F2(步调)人人 A>N>C(但这是场景混淆);F1(激进度)连人类都几乎持平 → ANC 标签没编码激进度维度",
             fontsize=12.5,fontweight="bold",y=1.02)
plt.tight_layout()
out="youdrive/metrics/08_anc_stratified.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
