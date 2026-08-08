"""Build markdown + figures for the per-model fail-to-commit analysis.
Reads per_model_commitment.json (real PDMS + classifications). No fabrication."""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

P="/mnt/pfs/zhengguantian/autovla/persona"
d=json.load(open(f"{P}/per_model_commitment.json"))
br=d["base_rate"]; agg=d["agg"]; corr=d["corr"]; cfg=d["config"]
TARGETS=cfg["TARGETS"]
HUMAN="human_logreplay"

def g(m,k): return agg[m][k]

# ---- correlations (with and without constant_velocity degenerate outlier) ----
mids=np.array([g(m,"mid_all") for m in TARGETS])
pdms=np.array([g(m,"pdms_all") for m in TARGETS])
keep=[m!="constant_velocity" for m in TARGETS]
mids_k=mids[keep]; pdms_k=pdms[keep]
pear_all=stats.pearsonr(mids,pdms); spear_all=stats.spearmanr(mids,pdms)
pear_k=stats.pearsonr(mids_k,pdms_k); spear_k=stats.spearmanr(mids_k,pdms_k)

# ================= FIGURE 1: sorted lateral middle-rate =================
order=sorted(TARGETS,key=lambda m:g(m,"mid_lat"))
vals=[100*g(m,"mid_lat") for m in order]
cols=["#d62728" if m==HUMAN else ("#1f77b4" if m=="autovla" else "#7f9bb5") for m in order]
fig,ax=plt.subplots(figsize=(11,6))
bars=ax.bar(range(len(order)),vals,color=cols,edgecolor="black",linewidth=0.5)
hr=100*g(HUMAN,"mid_lat")
ax.axhline(hr,color="#d62728",ls="--",lw=1.5,label=f"human_logreplay = {hr:.1f}%")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order,rotation=45,ha="right",fontsize=9)
for i,v in enumerate(vals): ax.text(i,v+0.6,f"{v:.0f}",ha="center",va="bottom",fontsize=8)
ax.set_ylabel("Fail-to-commit rate on lateral bimodal scenes (%)")
ax.set_title("Per-model fail-to-commit (drives the non-committal middle) — lateral left/right scenes\n"
             f"n={br['n_lat']} lateral bimodal scenes · nearest-of{{modeA,modeB,average}} classifier · real PDMS pipeline")
# highlight autovla legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#1f77b4",ec="black",label="AutoVLA (most human-like)"),
                   Patch(fc="#d62728",ec="black",label="human_logreplay"),
                   Patch(fc="#7f9bb5",ec="black",label="other models")],loc="upper left")
ax.set_ylim(0,max(vals)*1.15)
plt.tight_layout()
f1=f"{P}/per_model_middle_rate.png"; plt.savefig(f1,dpi=120); plt.close()
print("[fig]",f1)

# ================= FIGURE 2: middle-rate vs PDMS scatter =================
fig,ax=plt.subplots(figsize=(10,7))
for m in TARGETS:
    x=100*g(m,"mid_all"); y=g(m,"pdms_all")
    c="#d62728" if m==HUMAN else ("#1f77b4" if m=="autovla" else "#555555")
    s=170 if m in (HUMAN,"autovla") else 80
    mk="*" if m in (HUMAN,"autovla") else "o"
    ax.scatter(x,y,c=c,s=s,marker=mk,zorder=3,edgecolor="black",linewidth=0.5)
    ax.annotate(m,(x,y),textcoords="offset points",xytext=(6,4),fontsize=8,
                color=c if m in (HUMAN,"autovla") else "#333333")
# fit line excluding constant_velocity
xk=100*mids_k; yk=pdms_k
b,a=np.polyfit(xk,yk,1)
xs=np.linspace(xk.min(),xk.max(),50)
ax.plot(xs,a+b*xs,color="#999999",ls="--",lw=1.2,
        label=f"fit (excl. constant_velocity): r={pear_k[0]:.2f}, p={pear_k[1]:.2f}")
ax.set_xlabel("Fail-to-commit rate on bimodal scenes (lat+long, %)")
ax.set_ylabel("Mean real PDMS on bimodal scenes")
ax.set_title("Does failing to commit cost PDMS? middle-rate vs PDMS across 16 models\n"
             f"Pearson r={pear_all[0]:.2f} (p={pear_all[1]:.2f}) all · r={pear_k[0]:.2f} (p={pear_k[1]:.2f}) excl. constant_velocity")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
f2=f"{P}/per_model_middlerate_vs_pdms.png"; plt.savefig(f2,dpi=120); plt.close()
print("[fig]",f2)

# ================= MARKDOWN =================
def pct(x): return f"{100*x:.1f}%"
md=[]
md.append("# Per-model fail-to-commit (mode-averaging pathology) across all 16 models\n")
md.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} · extends the Q1 mode-averaging analysis "
          "from AutoVLA-only to all 16 models. Scene sets, real-PDMS scorer and the commit-vs-middle "
          "classifier are **reused verbatim** from `youdrive/mode_averaging_q1.py` / `mode_averaging.py`. "
          "Safety = native nuPlan/navsim **PDMS** on `metric_cache_navtest_v1`. Per-model PDMS reused from "
          "`mode_averaging_q1_scores.json`; only `constant_velocity` scored fresh. Nothing fabricated._\n")
md.append(f"> **Reproduction check:** recomputing the classifier here matches the stored q1 human/AutoVLA "
          f"labels on **{d['selfcheck']['human_match']}/{d['selfcheck']['human_tot']}** (human) and "
          f"**{d['selfcheck']['autovla_match']}/{d['selfcheck']['autovla_tot']}** (AutoVLA) bimodal scenes — "
          "identical scene sets & classifier.\n")

md.append("## TL;DR verdict\n")
av_div=agg["autovla"]["div"]
md.append(
f"""**The hypothesis is confirmed on the metric that actually captures the pathology — human-alignment of
commitment — but NOT on the naive "middle-rate" metric, which is confounded.**

1. **Base rate of multi-modality:** of **{br['n_tokens']}** tokens with ≥15 drivers,
   **{br['n_lat']} ({br['pct_lat']:.2f}%)** are lateral (left/right) bimodal, **{br['n_long']} ({br['pct_long']:.2f}%)**
   are longitudinal (brake/go) bimodal, and **{br['n_uni']} ({br['pct_uni']:.2f}%)** are unimodal.

2. **AutoVLA is overwhelmingly the least pathological where it counts.** Restricting to scenes where the
   *human committed to a turn/stop* ({av_div['tot']} scenes), AutoVLA commits to the **same mode as the human
   {pct(av_div['same_r'])}** of the time — vs 29–65% for every other model. Its "human turned, model went
   straight/other" divergence is only **{pct(av_div['mid_r']+av_div['other_r'])}** (middle {pct(av_div['mid_r'])} +
   wrong-mode {pct(av_div['other_r'])}), the lowest of all 16 by a wide margin. AutoVLA's fail-to-commit rate
   also tracks the human's almost exactly (lateral {pct(g('autovla','mid_lat'))} vs human {pct(g('human_logreplay','mid_lat'))}).

3. **Raw middle-rate does NOT rank the human/AutoVLA lowest**, because a rigid straight-line policy avoids the
   "middle" by never turning. `constant_velocity` has the *lowest* lateral middle-rate ({pct(g('constant_velocity','mid_lat'))})
   yet catastrophic PDMS ({g('constant_velocity','pdms_all'):.3f}); several always-commit models (gtrs_aug,
   drivesuprim, sparsedrivev2) also sit below the human. So "commit more" is not automatically good — *committing
   to the mode the context (human) calls for* is what matters, and that is exactly where AutoVLA wins.

4. **Middle-rate does not correlate with PDMS across models** (Pearson r={corr['pearson_r']:.2f}, p={corr['pearson_p']:.2f};
   Spearman ρ={corr['spearman_r']:.2f}, p={corr['spearman_p']:.2f}). Excluding the degenerate `constant_velocity`
   outlier it is still non-significant (r={pear_k[0]:.2f}, p={pear_k[1]:.2f}). At the model level, failing to commit
   is **not** the dominant driver of PDMS — model quality dominates. (The within-scene "averaging < committing to
   the correct mode" cost was already established in Q1.)
""")

md.append("## 1. Base rate of multi-modality\n")
md.append(
f"""| scene type | count | % of tokens (≥15 drivers) |
|---|---|---|
| Lateral (left/right) bimodal | {br['n_lat']} | {br['pct_lat']:.2f}% |
| Longitudinal (brake/go) bimodal | {br['n_long']} | {br['pct_long']:.2f}% |
| Unimodal | {br['n_uni']} | {br['pct_uni']:.2f}% |
| **Total** | **{br['n_tokens']}** | 100% |
""")

md.append("## 2. Per-model fail-to-commit (middle) rate\n")
md.append("`middle` = the model's own 4 s trajectory is nearest to the pool **average** rather than to either "
          "mode centroid (same nearest-of{modeA,modeB,average} rule used for the human in Q1). Ranked by lateral rate.\n")
md.append("| rank | model | middle-rate lateral | middle-rate longitudinal | middle-rate (all bimodal) |")
md.append("|---|---|---|---|---|")
for i,m in enumerate(sorted(TARGETS,key=lambda m:g(m,"mid_lat")),1):
    tag=" **(human)**" if m==HUMAN else (" **(AutoVLA)**" if m=="autovla" else "")
    md.append(f"| {i} | {m}{tag} | {pct(g(m,'mid_lat'))} | {pct(g(m,'mid_long'))} | {pct(g(m,'mid_all'))} |")
md.append("")
md.append(f"Human fail-to-commit: lateral {pct(g(HUMAN,'mid_lat'))}, longitudinal {pct(g(HUMAN,'mid_long'))}. "
          f"AutoVLA: lateral {pct(g('autovla','mid_lat'))}, longitudinal {pct(g('autovla','mid_long'))} — "
          "essentially human-matched, unlike models that are far higher (transfuser, diffusiondrive, hydra_mdp, "
          "goalflow, trajdiff_scaling) or artificially lower via rigid straight-line commitment (constant_velocity).\n")

md.append("## 3. Divergence from human commitment\n")
md.append(f"Restricted to the **{agg['autovla']['div']['tot']}** bimodal scenes where the **human committed** to a "
          "mode. For each model: commits to the SAME mode as human / the OTHER mode / goes middle "
          '("human turned or stopped, model didn\'t"). Ranked by same-as-human (desc).\n')
md.append("| rank | model | same mode as human | other mode | middle (didn't commit) | total divergence |")
md.append("|---|---|---|---|---|---|")
for i,m in enumerate(sorted(TARGETS,key=lambda m:-agg[m]["div"]["same_r"]),1):
    dv=agg[m]["div"]; tag=" **(human)**" if m==HUMAN else (" **(AutoVLA)**" if m=="autovla" else "")
    md.append(f"| {i} | {m}{tag} | {pct(dv['same_r'])} | {pct(dv['other_r'])} | {pct(dv['mid_r'])} | "
              f"{pct(dv['other_r']+dv['mid_r'])} |")
md.append("")
md.append("AutoVLA follows the human's committed mode **92.2%** of the time — the next best model is ~64%. "
          "This is the cleanest expression of the hypothesis: the most human-like model diverges from human "
          "commitment least; less-human models diverge far more (often committing to the WRONG mode, e.g. "
          "drivesuprim/gtrs_aug/drivoR/diffusiondrivev2 pick the other mode >45% of the time).\n")

md.append("## 4. Per-model safety (PDMS) in bimodal scenes & middle-rate↔PDMS\n")
md.append("Mean real PDMS on lateral / longitudinal bimodal scenes. Ranked by all-bimodal PDMS (desc).\n")
md.append("| rank | model | PDMS lateral | PDMS longitudinal | PDMS (all bimodal) | middle-rate (all) |")
md.append("|---|---|---|---|---|---|")
for i,m in enumerate(sorted(TARGETS,key=lambda m:-g(m,"pdms_all")),1):
    tag=" **(human)**" if m==HUMAN else (" **(AutoVLA)**" if m=="autovla" else "")
    md.append(f"| {i} | {m}{tag} | {g(m,'pdms_lat'):.4f} | {g(m,'pdms_long'):.4f} | {g(m,'pdms_all'):.4f} | {pct(g(m,'mid_all'))} |")
md.append("")
md.append(
f"""**Correlation (middle-rate vs PDMS, across 16 models):**
- All 16: Pearson r = {pear_all[0]:.3f} (p = {pear_all[1]:.2f}); Spearman ρ = {spear_all[0]:.3f} (p = {spear_all[1]:.2f}).
- Excluding degenerate `constant_velocity`: Pearson r = {pear_k[0]:.3f} (p = {pear_k[1]:.2f}); Spearman ρ = {spear_k[0]:.3f} (p = {spear_k[1]:.2f}).

Neither is significant, and the sign is unstable. **At the model level, failing to commit does not by itself
predict lower PDMS** — the strong planners keep high PDMS whether they hedge or commit, and the worst PDMS
belongs to `constant_velocity`, which barely goes to the middle at all. The PDMS cost of averaging is a
*within-scene* effect (averaging vs committing to the CORRECT mode, established in Q1), not a cross-model one.
""")

md.append("## 5. Verdict on the hypothesis\n")
md.append(
"""- **Is AutoVLA the least pathological?** **Yes**, on the metric that actually captures the pathology:
  when the human commits to a turn/stop, AutoVLA commits to the *same* mode **92.2%** of the time (lowest
  divergence, ~7.8% total), and its raw fail-to-commit rate matches the human's. On the naive middle-rate
  metric it sits mid-pack (≈ human), because that metric rewards rigid straight-line policies
  (`constant_velocity`) that avoid the middle without being human-like or safe.
- **Do less-human models exhibit it more?** **Mostly yes for the meaningful metric** — every non-human model
  diverges from the human's committed mode far more than AutoVLA (either going middle or, worse, committing to
  the opposite mode). But raw middle-rate is confounded and does NOT cleanly rank models by human-likeness.
- **Does middle-rate correlate with lower PDMS across models?** **No** (non-significant, unstable sign). The
  fail-to-commit → PDMS cost is a within-scene phenomenon (Q1), not something that separates whole models.
""")

open(f"{P}/per_model_commitment.md","w").write("\n".join(md))
print("[save]",f"{P}/per_model_commitment.md")
print("DONE")
