"""Heatmap of all computable dimensions, each model normalized to HUMAN (ratio)."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import TwoSlopeNorm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
D=json.load(open("youdrive/extended_metrics_out.json")); L=D["labels"]
models=["autovla","diffusiondrive","diffusiondrivev2","transfuser"]
KIN=["v_avg","v_std","peak_accel","peak_decel","long_jerk_rms","lat_amax","lat_jerk_rms","yaw_rate_max"]
ENVK=["thw_min","ttc_min","lead_gap_min","dist_any_min","dist_vru_min"]
dims=KIN+ENVK
M=np.zeros((len(dims),len(models)))
for i,k in enumerate(dims):
    grp="kin" if k in KIN else "env"
    h=D[grp]["human"][k]
    for j,m in enumerate(models):
        M[i,j]=D[grp][m][k]/h if h else np.nan
fig,ax=plt.subplots(figsize=(9,9))
norm=TwoSlopeNorm(vmin=0.3,vcenter=1.0,vmax=3.0)
im=ax.imshow(np.clip(M,0.3,3.0),cmap="RdYlGn_r",norm=norm,aspect="auto")
ax.set_xticks(range(len(models))); ax.set_xticklabels(models,rotation=20,ha="right")
ax.set_yticks(range(len(dims))); ax.set_yticklabels([L[k] for k in dims])
for i in range(len(dims)):
    for j in range(len(models)):
        ax.text(j,i,f"{M[i,j]:.2f}×",ha="center",va="center",fontsize=9,
                color="white" if (M[i,j]>2 or M[i,j]<0.5) else "black")
ax.axhline(len(KIN)-0.5,color="black",lw=2)
ax.text(-0.6,(len(KIN)-1)/2,"自车\n动力学",rotation=90,va="center",ha="center",fontsize=10,fontweight="bold")
ax.text(-0.6,len(KIN)+(len(ENVK)-1)/2,"环境\n交互",rotation=90,va="center",ha="center",fontsize=10,fontweight="bold")
plt.colorbar(im,label="相对人类倍数 (1.0=和人类一样, >1 红=更激进/更猛, <1 绿=更温和)")
ax.set_title("各维度相对人类的倍数(4049场景中位数)\n红=偏离人类(更猛/更挤),绿=更温和;越接近1.0越像人",
             fontsize=12,fontweight="bold")
plt.tight_layout(); out="youdrive/metrics/11_dims_vs_human.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
