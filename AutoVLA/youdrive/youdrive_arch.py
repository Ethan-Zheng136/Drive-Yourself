"""YouDrive on AutoVLA — overall architecture diagram with Style-LoRA as the style knob."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_fp="/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(17, 10.2)); ax.set_xlim(0,17); ax.set_ylim(0,10.2); ax.axis("off")

def box(x,y,w,h,t,fc,ec="#333",fs=10,bold=False,tc="#111"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",
                fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal",color=tc,wrap=True)

def arrow(x1,y1,x2,y2,c="#444",lw=1.8,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,
                color=c,lw=lw,ls=ls,shrinkA=2,shrinkB=2))

# ===== title
ax.text(8.5,9.85,"YouDrive on AutoVLA — Style-LoRA 作为风格旋钮",ha="center",fontsize=16,fontweight="bold")
ax.text(8.5,9.45,"灰=AutoVLA 原生(冻结)   橙=YouDrive 新增   红=风格控制信号 α",ha="center",fontsize=10,color="#555")

# ===== INPUT column
box(0.3,6.6,2.6,1.0,"多视角相机\n(front/L/R ...)","#dbeafe",fs=10)
box(0.3,5.2,2.6,0.9,"Ego 状态\n(速度/加速度/朝向)","#dbeafe",fs=10)
box(0.3,3.8,2.6,0.9,"导航指令\n(左转/直行/右转)","#dbeafe",fs=10)

# ===== Vision encoder
box(3.5,6.3,2.4,1.4,"Qwen2.5-VL\nVision Encoder\n(冻结)","#e5e7eb",fs=10,bold=True)
arrow(2.9,7.1,3.5,7.0); arrow(2.9,5.65,3.5,6.7)

# visual + text tokens
box(6.4,6.3,2.0,1.4,"多模态\ntoken 序列\n(visual+text)","#f3f4f6",fs=10)
arrow(5.9,7.0,6.4,7.0)
arrow(2.9,4.25,6.4,6.6,c="#777",ls="--")  # nav cmd as text prompt

# ===== LLM backbone (frozen) with LoRA
lx,ly,lw,lh=9.1,4.3,3.6,3.6
ax.add_patch(FancyBboxPatch((lx,ly),lw,lh,boxstyle="round,pad=0.05,rounding_size=0.15",
            fc="#eceff3",ec="#333",lw=1.8))
ax.text(lx+lw/2,ly+lh-0.32,"Qwen2.5-VL LLM backbone (冻结 ❄)",ha="center",fontsize=10.5,fontweight="bold")
# transformer layers
for i in range(3):
    yy=ly+0.55+i*0.95
    box(lx+0.25,yy,1.7,0.7,f"Transformer\nblock",fc="#dfe3e8",fs=8.5)
    # LoRA tap
    box(lx+2.15,yy,1.25,0.7,"Style\nLoRA","#fdba74",ec="#c2410c",fs=8.5,bold=True)
    ax.add_patch(FancyArrowPatch((lx+1.95,yy+0.35),(lx+2.15,yy+0.35),arrowstyle="-|>",
                mutation_scale=10,color="#c2410c",lw=1.4))
    ax.add_patch(FancyArrowPatch((lx+2.78,yy+0.0),(lx+1.1,yy-0.25),arrowstyle="-|>",
                mutation_scale=9,color="#c2410c",lw=1.0,ls=":",connectionstyle="arc3,rad=-0.3"))
arrow(8.4,7.0,9.1,6.5)

# ===== Action decoder + codebook
box(13.2,5.8,2.4,1.3,"自回归\nAction-token 解码\n(冻结)","#e5e7eb",fs=9.5,bold=True)
arrow(12.7,6.1,13.2,6.45)
box(13.2,4.0,2.4,1.1,"Action Codebook\n(运动基元词表)","#f3f4f6",fs=9.5)
arrow(14.4,5.8,14.4,5.1)
box(13.2,2.3,2.4,1.1,"轨迹 (x,y,θ)\nNavSim 评测","#dcfce7",ec="#15803d",fs=9.5,bold=True)
arrow(14.4,4.0,14.4,3.4)

# ===== Style control knob (the star)
ax.add_patch(FancyBboxPatch((9.0,0.5),7.2,2.9,boxstyle="round,pad=0.06,rounding_size=0.15",
            fc="#fff7ed",ec="#c2410c",lw=2.0))
ax.text(12.6,3.1,"★ 风格旋钮:Task Arithmetic 合成 LoRA",ha="center",fontsize=11.5,fontweight="bold",color="#c2410c")
box(9.3,1.6,1.5,0.9,"v_A\n(激进)","#fecaca",ec="#b91c1c",fs=9)
box(11.0,1.6,1.5,0.9,"v_N\n(中性)","#fde68a",ec="#a16207",fs=9)
box(12.7,1.6,1.5,0.9,"v_C\n(保守)","#bbf7d0",ec="#15803d",fs=9)
ax.text(14.9,2.05,"δ(α)=δ_avg\n+ Σ α_s · v_s",ha="center",va="center",fontsize=10,fontweight="bold",color="#7c2d12",
        bbox=dict(boxstyle="round",fc="white",ec="#c2410c"))
ax.text(12.6,0.85,"α = 连续风格刻度 (用户/上层策略给定) → 同一冻结模型输出不同风格轨迹",
        ha="center",fontsize=9.5,color="#7c2d12")
# knob feeds LoRA
ax.add_patch(FancyArrowPatch((11.7,3.4),(11.2,4.3),arrowstyle="-|>",mutation_scale=16,
            color="#dc2626",lw=2.4))
ax.text(11.0,3.85,"α",color="#dc2626",fontsize=13,fontweight="bold")

# ===== legend bottom-left
leg=[Line2D([0],[0],color="#9ca3af",lw=8,label="AutoVLA 原生·冻结"),
     Line2D([0],[0],color="#fdba74",lw=8,label="YouDrive Style-LoRA(可训练)"),
     Line2D([0],[0],color="#dc2626",lw=3,label="风格控制信号 α")]
ax.legend(handles=leg,loc="lower left",fontsize=9.5,frameon=True)

# training note
ax.text(0.3,2.6,"训练:仅训练 Style-LoRA(其余全冻结)\n每种风格单独蒸馏一组 v_s\n推理:按 α 在权重空间线性组合\n→ 无需重训即可连续调风格",
        fontsize=9.5,color="#334155",va="top",
        bbox=dict(boxstyle="round,pad=0.4",fc="#f1f5f9",ec="#94a3b8"))

out="/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/youdrive_arch.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("saved",out)
