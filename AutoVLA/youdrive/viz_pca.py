"""PCA loadings heatmap: how raw metrics group into composite style factors."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
P=json.load(open("youdrive/pca_style_out.json"))
feats=list(P["loadings"]); L=np.array([P["loadings"][f] for f in feats])  # [F,3]
LAB={"v_avg":"平均速度","v_std":"速度波动","peak_acc":"峰值加速","peak_dec":"峰值刹车",
     "long_jerk":"纵向jerk","lat_amax":"横向加速度","lat_jerk":"横向jerk",
     "thw_min":"THW","ttc_min":"TTC","dist_any":"最近他车","dist_vru":"最近VRU"}
fig,(ax,axb)=plt.subplots(1,2,figsize=(15,6),gridspec_kw={"width_ratios":[2.2,1]})
im=ax.imshow(L,cmap="RdBu",vmin=-0.6,vmax=0.6,aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["PC1\n动力学/平顺\n(28%)","PC2\n步调(速度vs刹车)\n(15%)","PC3\n社交裕度\n(13%)"],fontsize=10)
ax.set_yticks(range(len(feats))); ax.set_yticklabels([LAB[f] for f in feats])
for i in range(len(feats)):
    for j in range(3): ax.text(j,i,f"{L[i,j]:.2f}",ha="center",va="center",fontsize=9,
        color="white" if abs(L[i,j])>0.4 else "black")
plt.colorbar(im,ax=ax,label="载荷(权重)")
ax.set_title("PCA 载荷:11个原始量 → 3个独立风格因子\n(人类2772场景;同色同号=同一因子)",fontsize=12,fontweight="bold")
evr=P["evr"]; axb.bar(range(1,len(evr)+1),[e*100 for e in evr],color="#4363d8")
axb.plot(range(1,len(evr)+1),np.cumsum(evr)*100,"o-",color="#e6194B")
axb.set_xlabel("主成分"); axb.set_ylabel("方差解释%"); axb.set_title("方差解释(红=累计)\n前3因子=56%,无单一主导轴",fontsize=11,fontweight="bold")
axb.axhline(0); axb.grid(alpha=0.3)
plt.tight_layout(); out="youdrive/metrics/12_pca_factors.png"; plt.savefig(out,dpi=120,bbox_inches="tight"); print("saved",out)
