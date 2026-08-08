"""Render divergence_safety_link.md + .png from divergence_safety_link.json.
Real data only; reads the analysis JSON produced by divergence_safety_link.py."""
import json, numpy as np
PERSONA="/mnt/pfs/zhengguantian/autovla/persona"
J=json.load(open(f"{PERSONA}/divergence_safety_link.json"))
OUT_MD=f"{PERSONA}/divergence_safety_link.md"; OUT_PNG=f"{PERSONA}/divergence_safety_link.png"

pm=J["per_model"]; pooled=J["pooled"]; pex=J["pooled_excl_cv"]
pa=J["paired_all"]; pax=J["paired_excl_cv"]; hr=J["human_ref"]
NON=J["config"]["NONHUMAN"]; n_hc=J["human_committed_scenes"]

def f(x,nd=3):
    return "n/a" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.{nd}f}"
def pf(p):
    if p is None: return "n/a"
    return f"{p:.1e}" if p<1e-3 else f"{p:.3f}"

# ---------------- figure: 2 panels ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.5,5.4))

# panel 1: pooled matched / wrong / middle (as requested) + human ref
cats=["Matched\nhuman mode","Wrong\nmode","Middle\n(no commit)"]
vals=[pooled["pdms_matched"],pooled["pdms_wrong"],pooled["pdms_middle"]]
ns=[pooled["n_matched"],pooled["n_wrong"],pooled["n_middle"]]
colors=["#2e7d32","#c62828","#ef6c00"]
bars=ax1.bar(cats,vals,color=colors,width=0.62,edgecolor="black",linewidth=0.6)
for b,v,n in zip(bars,vals,ns):
    ax1.text(b.get_x()+b.get_width()/2,v+0.008,f"{v:.3f}\n(n={n})",ha="center",va="bottom",fontsize=10)
hy=hr["pdms_mean"]
ax1.axhline(hy,ls="--",color="#1565c0",lw=1.6)
ax1.text(2.48,hy+0.005,f"Human ref = {hy:.3f}",color="#1565c0",ha="right",va="bottom",fontsize=10)
ax1.set_ylabel("Mean real PDMS")
ax1.set_ylim(0,max(max(vals),hy)*1.16)
ax1.grid(axis="y",ls=":",alpha=0.5)
ax1.set_title("(a) Pooled by model's own behavior\n(observation-level; confounded by model mix)",fontsize=11)
d=pooled["delta_div_minus_matched"]; p=pooled["p_matched_vs_diverged"]
ax1.text(0.02,0.02,f"diverged - matched Δ={d:+.3f}  (MWU p={pf(p)})",transform=ax1.transAxes,fontsize=9,style="italic")

# panel 2: within-model paired matched vs diverged (clean test)
models=[m for m in NON if pm[m]["n_matched"]>=5 and pm[m]["n_diverged"]>=5]
mv=[pm[m]["pdms_matched"] for m in models]; dv=[pm[m]["pdms_diverged"] for m in models]
order=np.argsort(mv)
models=[models[i] for i in order]; mv=[mv[i] for i in order]; dv=[dv[i] for i in order]
y=np.arange(len(models))
for yi,a,b in zip(y,mv,dv):
    ax2.plot([a,b],[yi,yi],color="#9e9e9e",lw=1.4,zorder=1)
ax2.scatter(mv,y,color="#2e7d32",s=42,zorder=2,label="matched")
ax2.scatter(dv,y,color="#c62828",s=42,zorder=2,label="diverged")
ax2.set_yticks(y); ax2.set_yticklabels(models,fontsize=8)
ax2.set_xlabel("Model's own mean real PDMS")
ax2.axvline(hy,ls="--",color="#1565c0",lw=1.2)
ax2.grid(axis="x",ls=":",alpha=0.5)
ax2.legend(loc="lower left",fontsize=9,frameon=True)
ax2.set_title(f"(b) Within-model: matched vs diverged\n(paired Wilcoxon p={pf(pa['wilcoxon_p'])}, meanΔ={pa['mean_delta']:+.3f})",fontsize=11)
fig.suptitle(f"Divergence-from-human safety link  |  {n_hc} human-committed scenes  |  {len(NON)} non-human models",fontsize=12.5,y=1.00)
fig.tight_layout()
fig.savefig(OUT_PNG,dpi=120,bbox_inches="tight")
print("[png]",OUT_PNG)

# ---------------- markdown ----------------
L=[]
L.append("# Divergence-from-human safety link (real PDMS)\n")
L.append("**Question.** Within the 652 scenes where the *human* committed to a maneuver mode, "
         "does a model's own real PDMS drop on scenes where it **diverged** from the human's mode "
         "(chose the wrong real mode, or drove the non-committal middle) versus scenes where it "
         "**matched** the human's mode? This is a strict **within-model, within-human-committed-scene** "
         "comparison — it does *not* use the (already shown to be confounded) cross-model "
         "middle-rate↔PDMS correlation.\n")
L.append("**Method (reused artifacts).** Bimodal scene set + 3-way commit classifier "
         "(nearest of {mode A, mode B, pool-average middle} on the 8-pt canonical trajectory) "
         "and native navsim PDMS on `metric_cache_navtest_v1` are identical to "
         "`per_model_commitment.py` / `mode_averaging_q1.py`. Per-model PDMS reused from "
         "`mode_averaging_q1_scores.json`; `constant_velocity` scored fresh with the same "
         "`PDM_Reward`→`pdm_score` path. 16 target models (15 non-human + human reference); "
         "652 of 861 bimodal scenes had a committed human.\n")

L.append("## Headline — the safety link does NOT hold on average\n")
L.append("| Test | matched PDMS | diverged PDMS | Δ (diverged − matched) | p |")
L.append("|---|---|---|---|---|")
L.append(f"| **Pooled, observation-level** (all 15 non-human models) | {f(pooled['pdms_matched'])} | {f(pooled['pdms_diverged'])} | **{pooled['delta_div_minus_matched']:+.3f}** | {pf(pooled['p_matched_vs_diverged'])} (Mann-Whitney) |")
L.append(f"| Pooled, excl. `constant_velocity` | {f(pex['pdms_matched'])} | {f(pex['pdms_diverged'])} | {pex['delta_div_minus_matched']:+.3f} | {pf(pex['p_matched_vs_diverged'])} |")
L.append(f"| **Within-model paired** (Wilcoxon, {pa['n_models']} models) | {f(pa['mean_matched'])} | {f(pa['mean_diverged'])} | **{pa['mean_delta']:+.3f}** | {pf(pa['wilcoxon_p'])} |")
L.append(f"| Within-model paired, excl. `constant_velocity` ({pax['n_models']} models) | {f(pax['mean_matched'])} | {f(pax['mean_diverged'])} | {pax['mean_delta']:+.3f} | {pf(pax['wilcoxon_p'])} |")
L.append("")
L.append("Every clean test is **null**: the within-model paired mean Δ is ≈ 0 "
         f"({pa['mean_delta']:+.4f}, {pa['n_delta_negative']}/{pa['n_models']} models negative, "
         f"Wilcoxon p={pf(pa['wilcoxon_p'])}). The naive observation-pooled comparison even shows "
         "diverged *slightly higher* (Δ = +0.027), but that is exactly the cross-model confound the "
         "within-model design was meant to remove — pooling mixes low-PDMS models (esp. "
         "`constant_velocity`, PDMS≈0.23) whose matched scenes drag the pooled *matched* mean down. "
         "**Conclusion: failing to follow the human's committed maneuver does not, in general, make a "
         "model less safe by real PDMS.**\n")

L.append("## Wrong-mode vs middle breakdown (pooled)\n")
L.append("| Partition | mean PDMS | n obs |")
L.append("|---|---|---|")
L.append(f"| Matched human mode | {f(pooled['pdms_matched'])} | {pooled['n_matched']} |")
L.append(f"| Wrong mode (other real maneuver) | {f(pooled['pdms_wrong'])} | {pooled['n_wrong']} |")
L.append(f"| Middle (mode-averaging / no commit) | {f(pooled['pdms_middle'])} | {pooled['n_middle']} |")
L.append("")
L.append(f"Pooled contrasts: matched vs wrong p={pf(pooled['p_matched_vs_wrong'])}; "
         f"matched vs middle p={pf(pooled['p_matched_vs_middle'])}; "
         f"wrong vs middle p={pf(pooled['p_wrong_vs_middle'])}. "
         "The **middle** (mode-averaged) trajectory is *not* the least safe — pooled it is actually the "
         "highest of the three, because the geometric pool-average is smooth/central and scores well on "
         "PDMS. **Wrong-mode** is the lowest of the three but still not below matched at the pooled level. "
         "So 'diverged' as a whole is diluted: the genuinely-dangerous wrong-mode is offset by the "
         "safe-by-construction middle.\n")

L.append("## Per-model table\n")
L.append("| Model | n matched | n wrong | n middle | PDMS matched | PDMS wrong | PDMS middle | PDMS diverged | Δ (div−match) | p (match vs div) |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for m in NON:
    r=pm[m]
    L.append(f"| {m} | {r['n_matched']} | {r['n_wrong']} | {r['n_middle']} | {f(r['pdms_matched'])} | "
             f"{f(r['pdms_wrong'])} | {f(r['pdms_middle'])} | {f(r['pdms_diverged'])} | "
             f"{r['delta_div_minus_matched']:+.3f} | {pf(r['p_matched_vs_diverged'])} |")
L.append("")
# significant supporters (negative delta = diverged less safe = supports hypothesis)
sup=[m for m in NON if pm[m]['p_matched_vs_diverged'] is not None and pm[m]['p_matched_vs_diverged']<0.05 and pm[m]['delta_div_minus_matched']<0]
opp=[m for m in NON if pm[m]['p_matched_vs_diverged'] is not None and pm[m]['p_matched_vs_diverged']<0.05 and pm[m]['delta_div_minus_matched']>0]
L.append(f"**Where the link IS significant and in the hypothesized direction (diverged < matched):** "
         f"{', '.join(sup) if sup else 'none'}. These are the strong scoring/retrieval planners — they earn "
         "their high PDMS precisely by committing to the correct maneuver, so diverging costs them "
         "0.04–0.07 PDMS (all p<0.005).\n")
L.append(f"**Significant in the OPPOSITE direction (diverged > matched):** {', '.join(opp) if opp else 'none'}.\n")
L.append("For most trajectory-generation models (autovla, diffusiondrive, diffusiondrivev2, goalflow, "
         "transfuser, trajdiff_scaling), **wrong-mode** PDMS is clearly below matched (e.g. autovla 0.68 vs "
         "0.83; goalflow 0.69 vs 0.81), i.e. taking the *wrong real maneuver* is dangerous — but their "
         "**middle** PDMS is ≥ matched, so their aggregate 'diverged' Δ is near zero or positive.\n")

L.append("## Human reference\n")
L.append(f"Human (log-replay) is matched by definition on these scenes. On the {hr['n_committed_scenes']} "
         f"human-committed scenes the human's mean real PDMS = **{f(hr['pdms_mean'])}** "
         "(dashed blue line in the figure) — higher than any model partition, consistent with the human "
         "being the safe committed reference.\n")

L.append("## Scene counts / sanity\n")
L.append(f"- Bimodal scenes: lateral {J['config']['n_lat']} + longitudinal {J['config']['n_long']} = "
         f"{J['config']['n_lat']+J['config']['n_long']}; human committed in **{n_hc}**.\n")
small=[m for m in NON if pm[m]['n_wrong']<10 or pm[m]['n_middle']<10]
L.append(f"- Per-model matched/diverged n are all ≥ 5 (all 15 models enter the paired test). Small "
         f"individual sub-cells (wrong or middle < 10 obs): {', '.join(small) if small else 'none'} "
         "(autovla has only 18 wrong / 33 middle → its per-model p is under-powered).\n")

L.append("## Verdict\n")
L.append("**The safety link is NOT established as a general within-model effect.** Pooled and "
         "within-model paired tests are all null (paired mean Δ ≈ −0.002 PDMS, p ≈ 0.85; ≈ 0 and p ≈ 0.95 "
         "after removing the degenerate `constant_velocity`). Diverging from the human's committed mode "
         "does **not** systematically reduce a model's real PDMS.\n")
L.append("Two honest caveats keep the *mode-averaging* story alive in restricted form:\n")
L.append("1. **Wrong-mode divergence is genuinely less safe** for most trajectory-generation models "
         "(large negative wrong-vs-matched gaps), and\n")
L.append("2. **for the strong scoring/retrieval planners (GTRS, GTRS-aug, Hydra-MDP, DriveSuprim) "
         "matching the human's mode is significantly safer** than diverging (p<0.005).\n")
L.append("But the non-committal **middle** trajectory is *not* penalised by PDMS (it is geometrically "
         "smooth), which cancels the wrong-mode penalty in the aggregate. So the clean answer is: "
         "**PDMS does not, by itself, punish diverging from the human's commitment** — it only punishes "
         "taking the *wrong committed maneuver*, and only for some models. The 'models diverge from human "
         "commitment' argument therefore cannot be closed with a simple within-model PDMS safety penalty.\n")

open(OUT_MD,"w").write("\n".join(L))
print("[md]",OUT_MD)
print("DONE")
