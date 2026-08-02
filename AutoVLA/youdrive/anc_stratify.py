"""ANC-stratified style: for each model, mean F1/F2 on scenes humans labeled A / N / C.
Tests whether a model's style tracks the human style label (it should, if it personalizes)."""
import json, math
import numpy as np
R=json.load(open("youdrive/style_metrics_full4.json"))
models=list(R["per_token"])
ORDER=["A","N","C"]

def grp(per):
    out={a:{"F1":[],"F2":[]} for a in ORDER}
    for v in per.values():
        a=v.get("anc")
        if a in out:
            if v["F1_assert"]==v["F1_assert"]: out[a]["F1"].append(v["F1_assert"])
            if v["F2_pace"]==v["F2_pace"]: out[a]["F2"].append(v["F2_pace"])
    return out

# counts
hg=grp(R["human_per_token"])
print("ANC scene counts (human):", {a:len(hg[a]["F1"]) for a in ORDER})
print()
print(f"{'model':<16} | " + " | ".join(f"{a}:F1   F2  " for a in ORDER))
print("-"*70)
def line(name,g):
    cells=[]
    for a in ORDER:
        f1=np.mean(g[a]["F1"]) if g[a]["F1"] else float("nan")
        f2=np.mean(g[a]["F2"]) if g[a]["F2"] else float("nan")
        cells.append(f"{f1:.2f} {f2:.2f}")
    print(f"{name:<16} | " + " | ".join(cells))
line("HUMAN-GT",hg)
for m in models: line(m,grp(R["per_token"][m]))

# key signal: does F1 increase A->C? (human aggressive scenes should have higher F1 than conservative)
print("\n--- monotonicity F1(A) > F1(N) > F1(C)?  (does style track human label) ---")
for name,per in [("HUMAN-GT",R["human_per_token"])]+[(m,R["per_token"][m]) for m in models]:
    g=grp(per)
    f1=[np.mean(g[a]["F1"]) if g[a]["F1"] else float("nan") for a in ORDER]
    delta=f1[0]-f1[2]  # A minus C
    mono="✓ A>C" if delta>0.02 else ("≈ flat" if abs(delta)<=0.02 else "✗ reversed")
    print(f"  {name:<16} F1: A={f1[0]:.2f} N={f1[1]:.2f} C={f1[2]:.2f}   A-C={delta:+.2f}  {mono}")

json.dump({"human":{a:{"F1":float(np.mean(hg[a]["F1"])),"F2":float(np.mean(hg[a]["F2"])),"n":len(hg[a]["F1"])} for a in ORDER},
           "models":{m:{a:{"F1":float(np.mean(grp(R['per_token'][m])[a]["F1"])),"F2":float(np.mean(grp(R['per_token'][m])[a]["F2"]))} for a in ORDER} for m in models}},
          open("youdrive/anc_stratify_out.json","w"),indent=2)
print("\nwritten youdrive/anc_stratify_out.json")
