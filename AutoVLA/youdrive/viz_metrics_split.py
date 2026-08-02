"""One-metric-per-figure visualizations + histogram linkage (percentile explainer).
Each PNG shows exactly ONE metric with its definition written on the figure.
"""
import json, math, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
sys.path.insert(0, os.path.dirname(__file__))
from style_metrics import build_reference, human_features, model_features, style_coords, F1_POS, F2_POS

_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False

OUT="youdrive/metrics"; os.makedirs(OUT, exist_ok=True)
R=json.load(open("youdrive/style_metrics_anc.json"))
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
ref,counts=build_reference(ST)
MC={"AutoVLA":"#e6194B","DiffusionDrive":"#4363d8","DiffusionDriveV2":"#3cb44b","TransFuser":"#f58231"}
models=list(R["per_token"]); toks=sorted(R["human_per_token"])

def def_box(ax,txt):
    ax.text(0.5,-0.22,txt,transform=ax.transAxes,ha="center",va="top",fontsize=10,
            color="#334155",bbox=dict(boxstyle="round,pad=0.5",fc="#f1f5f9",ec="#94a3b8"))

# ---------- Fig 1: elasticity (one metric) ----------
el=R["top_elastic_scenarios"]; names=list(el); vals=[el[n]["elasticity"] for n in names]; ns=[el[n]["n"] for n in names]
colors=["#15803d" if v>=0.19 else ("#f59e0b" if v>=0.12 else "#dc2626") for v in vals]
fig,ax=plt.subplots(figsize=(9,5.5)); y=np.arange(len(names))
ax.barh(y,vals,color=colors); ax.set_yticks(y); ax.set_yticklabels([f"{n} (n={c})" for n,c in zip(names,ns)],fontsize=10)
ax.invert_yaxis(); ax.set_xlabel("elasticity")
for i,v in enumerate(vals): ax.text(v+0.004,i,f"{v:.3f}",va="center",fontsize=9)
ax.set_title("指标①  场景风格弹性 elasticity",fontsize=13,fontweight="bold")
def_box(ax,"定义:某 scenario_type 下人类风格特征的离散度(IQR/range 取平均)。\n只有高弹性场景能区分风格。红=lane_following 占58%却最低 → 解释了为何模型轨迹全重合。")
plt.tight_layout(); plt.savefig(f"{OUT}/01_elasticity.png",dpi=120,bbox_inches="tight"); plt.close()

# ---------- Fig 2: F1 assertiveness (one metric) ----------
fig,ax=plt.subplots(figsize=(9,5))
f1={m:np.nanmean([v["F1_assert"] for v in R["per_token"][m].values()]) for m in models}
f1h=np.nanmean([v["F1_assert"] for v in R["human_per_token"].values()])
bars=list(f1)+["Human GT"]; ys=[f1[m] for m in models]+[f1h]
cols=[MC[m] for m in models]+["black"]
ax.bar(bars,ys,color=cols); ax.set_ylim(0,1); ax.axhline(0.5,color="#bbb",ls=":")
for i,v in enumerate(ys): ax.text(i,v+0.02,f"{v:.3f}",ha="center",fontsize=10)
ax.set_ylabel("F1 (百分位)"); ax.set_title("指标②  纵向激进度 F1 Assertiveness",fontsize=13,fontweight="bold")
def_box(ax,"定义:同场景人类分布里『减速猛/jerk大/加速猛』的平均百分位(0.5=人类中位)。\n越高越激进。可见 4 模型都>0.7,远高于人类GT,且 DDv2≈1.0(最猛刹)。")
plt.tight_layout(); plt.savefig(f"{OUT}/02_F1_assertiveness.png",dpi=120,bbox_inches="tight"); plt.close()

# ---------- Fig 3: F2 pace (one metric) ----------
fig,ax=plt.subplots(figsize=(9,5))
f2={m:np.nanmean([v["F2_pace"] for v in R["per_token"][m].values()]) for m in models}
f2h=np.nanmean([v["F2_pace"] for v in R["human_per_token"].values()])
ys=[f2[m] for m in models]+[f2h]
ax.bar(bars,ys,color=cols); ax.set_ylim(0,1); ax.axhline(0.5,color="#bbb",ls=":")
for i,v in enumerate(ys): ax.text(i,v+0.02,f"{v:.3f}",ha="center",fontsize=10)
ax.set_ylabel("F2 (百分位)"); ax.set_title("指标③  步调 F2 Pace",fontsize=13,fontweight="bold")
def_box(ax,"定义:『速度偏好+横向活动』的平均百分位。越高越快/越爱动。\n4 模型均≈0.94-0.96,人类GT 0.84,模型整体比人快。")
plt.tight_layout(); plt.savefig(f"{OUT}/03_F2_pace.png",dpi=120,bbox_inches="tight"); plt.close()

# ---------- Fig 4: coverage (one metric) ----------
fig,ax=plt.subplots(figsize=(9,5))
cov={m:R["models"][m]["coverage_hull"] for m in models}; covh=R["human_ref"]["coverage_hull"]
ys=[cov[m] for m in models]+[covh]
ax.bar(bars,ys,color=cols)
for i,v in enumerate(ys): ax.text(i,v+max(ys)*0.02,f"{v:.4f}",ha="center",fontsize=10)
ax.set_ylabel("凸包面积"); ax.set_title("指标④  风格覆盖 coverage",fontsize=13,fontweight="bold")
def_box(ax,"定义:模型在 (F1,F2) 平面上轨迹点的凸包面积 = 能张开的风格范围。\n所有模型都远小于人类GT(0.035)→ 现成模型风格单一。")
plt.tight_layout(); plt.savefig(f"{OUT}/04_coverage.png",dpi=120,bbox_inches="tight"); plt.close()

# ---------- Fig 5: homogeneity matrix (one metric) ----------
fig,ax=plt.subplots(figsize=(7,6))
M=np.zeros((len(models),len(models)))
for i,a in enumerate(models):
    for j,b in enumerate(models):
        ds=[]
        for t in toks:
            va,vb=R["per_token"][a].get(t),R["per_token"][b].get(t)
            if va and vb and not any(math.isnan(x) for x in (va["F1_assert"],va["F2_pace"],vb["F1_assert"],vb["F2_pace"])):
                ds.append(math.dist((va["F1_assert"],va["F2_pace"]),(vb["F1_assert"],vb["F2_pace"])))
        M[i,j]=np.mean(ds) if ds else 0
im=ax.imshow(M,cmap="RdYlGn_r",vmin=0,vmax=0.5)
ax.set_xticks(range(len(models))); ax.set_yticks(range(len(models)))
ax.set_xticklabels(models,rotation=30,ha="right"); ax.set_yticklabels(models)
for i in range(len(models)):
    for j in range(len(models)): ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",fontsize=10)
plt.colorbar(im,label="平均风格距离"); ax.set_title("指标⑤  跨模型同质度 homogeneity",fontsize=13,fontweight="bold")
ax.text(0.5,-0.28,"定义:每对模型在同一 token 的 (F1,F2) 平均欧氏距离。越小越雷同(绿)。\n全部<0.26 → 模型间几乎无风格差异,印证『现成模型蒸馏不出风格』。",
        transform=ax.transAxes,ha="center",va="top",fontsize=10,color="#334155",
        bbox=dict(boxstyle="round,pad=0.5",fc="#f1f5f9",ec="#94a3b8"))
plt.tight_layout(); plt.savefig(f"{OUT}/05_homogeneity.png",dpi=120,bbox_inches="tight"); plt.close()

# ---------- Fig 6: histogram linkage (percentile explainer for ONE token) ----------
tok=toks[0]; ctx=ST[tok].get("scenario_type","unknown")
# collect human raw feature arrays for this scenario_type
hr={}
for t,rec in ST.items():
    if rec.get("scenario_type")!=ctx: continue
    f=human_features(rec)
    if f: 
        for k,v in f.items(): hr.setdefault(k,[]).append(v)
feats=["v_avg","peak_decel","jerk_rms","lat_vmax"]
fig,axes=plt.subplots(2,2,figsize=(13,9))
for ax,k in zip(axes.ravel(),feats):
    arr=np.asarray([x for x in hr.get(k,[]) if not (isinstance(x,float) and math.isnan(x))],float)
    ax.hist(arr,bins=30,color="#cbd5e1",edgecolor="#94a3b8")
    for m in models:
        raw=R["per_token"][m][tok]["raw"].get(k)
        if raw is not None and not (isinstance(raw,float) and math.isnan(raw)):
            pct=(arr<raw).mean()
            ax.axvline(raw,color=MC[m],lw=2)
            ax.text(raw,ax.get_ylim()[1]*0.9,f"{m[:4]}\n{pct*100:.0f}%",color=MC[m],fontsize=8,ha="center")
    hraw=human_features(ST[tok]).get(k)
    if hraw is not None and not math.isnan(hraw):
        ax.axvline(hraw,color="black",lw=2.5,ls="--"); ax.text(hraw,ax.get_ylim()[1]*0.7,"GT",fontsize=9,fontweight="bold")
    ax.set_title(f"{k}",fontsize=11,fontweight="bold"); ax.set_xlabel(k); ax.set_ylabel("人类场景数")
fig.suptitle(f"直方图联动:百分位怎么来的(token={tok[:8]}, 场景={ctx})\n灰=该场景全部人类分布,竖线=各模型原始值,百分位=竖线左侧人类占比",
             fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f"{OUT}/06_percentile_linkage.png",dpi=120,bbox_inches="tight"); plt.close()

print("saved 6 figures to",OUT)
for f in sorted(os.listdir(OUT)): print("  ",f)
