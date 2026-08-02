"""Render ANC example scenes (BEV map+agents) with Human GT + each available model trajectory."""
import os, json, glob, tempfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg, numpy as np
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.bev import add_trajectory_to_bev_ax

DATA="/root/workspace/closed_loop/data/navsim"
CMP="/mnt/pfs/zhengguantian/autovla/compare_anc"
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
toks=json.load(open("/mnt/pfs/zhengguantian/autovla/compare/tokens_anc.json"))

MODELC={"autovla":"#e6194B","diffusiondrive":"#4363d8","diffusiondrivev2":"#3cb44b","transfuser":"#f58231"}
MODELS={m:json.load(open(f"{CMP}/{m}.json")) for m in MODELC if os.path.exists(f"{CMP}/{m}.json")}
print("available models:", list(MODELS))

sf=SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,max_scenes=None,log_names=None,tokens=toks)
loader=SceneLoader(data_path=Path(f"{DATA}/navsim_logs/test"),sensor_blobs_path=Path(f"{DATA}/sensor_blobs/test"),scene_filter=sf,sensor_config=SensorConfig.build_no_sensors())
print("loaded",len(loader.tokens))

GT=dict(line_color="black",line_color_alpha=1.0,line_width=3.0,line_style="--",marker="",marker_size=0,marker_edge_color="black",zorder=20)
tmp=tempfile.mkdtemp(); pngs={}
for tok in toks:
    try:
        scene=loader.get_scene_from_token(tok)
        fidx=scene.scene_metadata.num_history_frames-1
        fig,ax=plot_bev_frame(scene,fidx)
        # human GT (dashed black)
        add_trajectory_to_bev_ax(ax,scene.get_future_trajectory(),GT)
        # models
        for m,c in MODELC.items():
            if m in MODELS and tok in MODELS[m]:
                a=np.array(MODELS[m][tok])
                ax.plot(a[:,1],a[:,0],'-o',color=c,ms=2.5,lw=1.8)
        anc=ST.get(tok,{}); ax.set_title(f"ANC={anc.get('ANC_result','?')} v={anc.get('v_avg',0):.1f} {anc.get('scenario_type','')}",fontsize=10)
        p=f"{tmp}/{tok}.png"; fig.savefig(p,dpi=85,bbox_inches="tight"); plt.close(fig); pngs[tok]=p
    except Exception as e:
        print("skip",tok,repr(e))
# grid 2x3
fig,axes=plt.subplots(2,3,figsize=(20,11))
for i,tok in enumerate(toks):
    ax=axes[i//3][i%3]; ax.axis("off")
    if tok in pngs: ax.imshow(mpimg.imread(pngs[tok]))
# legend
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],color="black",ls="--",lw=3,label="Human GT")]+[Line2D([0],[0],color=c,lw=2,label=m) for m,c in MODELC.items() if m in MODELS]
fig.legend(handles=leg,loc="upper center",ncol=5,fontsize=12)
fig.suptitle("ANC scenes: Human GT (dashed) vs models  |  BEV map+agents",fontsize=14,fontweight="bold",y=0.99)
out="/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/anc_scene_compare.png"
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(out,dpi=110,bbox_inches="tight"); print("saved",out)
