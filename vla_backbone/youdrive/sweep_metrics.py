"""Sweep -> style metrics: auto-compute & SAVE persona style metrics for a dump dir.

Given an OUTDIR produced by dump_persona_sweep_8gpu.sh (containing
``persona_a{ALPHA}.json`` token->trajectory dumps + a ``_meta`` key), this script:

  1. computes (or reuses) the env_interact social-interaction features for every
     alpha dump (others' agents THW/TTC/distance vs the model's predicted ego
     trajectory) -- see youdrive/env_interact.py;
  2. computes, per alpha:
       - YDSP factors DAI / PAI / SAI  (percentile vs human; 50=human-like, >50 more
         aggressive)   -- exact PCA-loadings + human-percentile pipeline reused from
         youdrive/persona_compare_full.py / persona_compare_with_kl1.py;
       - kinematic medians v_avg, peak_acc, peak_dec, long_jerk, lat_jerk
         (per-scene median trajectory)  -- reused from youdrive/ydsp_kin_compare.py;
       - social medians thw_min, ttc_min, lead_gap_min, dist_any_min, dist_vru_min
         (median over valid scenes)  -- reused from youdrive/env_interact.py;
       - L2-distance-to-DDv2 in YDSP space (Euclidean over [DAI,PAI,SAI]; 0 = matches
         teacher style);
  3. includes base AutoVLA + DDv2 teacher + human rows as comparison anchors
     (anchors loaded from compare_full/{autovla,diffusiondrivev2}.json +
     youdrive/env_interact_out.json -- the same anchors the compare scripts use);
  4. SAVES   OUTDIR/style_metrics.json   (structured)
             OUTDIR/style_metrics.txt    (human-readable table)
             OUTDIR/style_metrics.png    (dose-response overlay plot; optional);

REPO RULE: every output goes under OUTDIR (which must be on PFS,
/mnt/pfs/zhengguantian/...). Nothing is written into the repo/workspace.

Usage:
  PY=/root/workspace/miniconda3/envs/autovla/bin/python
  # compute env_interact itself (needs NUPLAN env + navsim GT; CPU-only, no GPU):
  $PY youdrive/sweep_metrics.py --outdir /mnt/pfs/.../my_sweep

  # reuse a precomputed env_interact json (e.g. validation against big6ep):
  $PY youdrive/sweep_metrics.py --outdir /mnt/pfs/.../compare_persona_big6ep \
      --env-json /mnt/pfs/.../persona/env_interact_big6ep.json \
      --env-key-tmpl 'big6ep_a{alpha}'
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AV = os.path.dirname(HERE)
DT = 0.5

# ---- fixed anchors (same files the compare_* scripts use) ----
ST_PATH = os.environ.get("STYLETEST", "/root/workspace/tools/scorer/data/styledrive/styletest.json")
ENV1_PATH = os.environ.get("ANCHOR_ENV", os.path.join(HERE, "env_interact_out.json"))
LOAD_PATH = os.environ.get("PCA_LOADINGS", os.path.join(HERE, "pca_style_out.json"))
COMPARE_FULL = os.environ.get("COMPARE_FULL", "/mnt/pfs/zhengguantian/autovla/compare_full")
BASE_DUMP = os.path.join(COMPARE_FULL, "autovla.json")
DDV2_DUMP = os.path.join(COMPARE_FULL, "diffusiondrivev2.json")

# ---- teacher (comparison anchor) registry ----
# name -> (dump filename under COMPARE_FULL, env_interact_out key, display label).
# NOTE: ddv2 == diffusiondrivev2 (same anchor); it is the DEFAULT so that unset
# behaviour is byte-identical to the historical hard-coded DDv2 comparison.
TEACHER_ANCHORS = {
    "ddv2": ("diffusiondrivev2.json", "diffusiondrivev2", "DDv2"),
    "transfuser": ("transfuser.json", "transfuser", "transfuser"),
    "goalflow": ("goalflow.json", "goalflow", "goalflow"),
    "gtrs": ("gtrs.json", "gtrs", "GTRS"),
    "hydra_mdp": ("hydra_mdp.json", "hydra_mdp", "Hydra-MDP"),
    "wote": ("wote.json", "wote", "WoTE"),
}

# YDSP factor feature layout (must match pca_style_out.json loadings order)
KIN = ["v_avg", "v_std", "peak_acc", "peak_dec", "long_jerk", "lat_amax", "lat_jerk"]
ENVK = ["thw_min", "ttc_min", "dist_any", "dist_vru"]            # loading feature names
ENVK_SRC = ["thw_min", "ttc_min", "dist_any_min", "dist_vru_min"]  # env_interact output names
ALLF = KIN + ENVK
ORI = np.array([-1, 1, -1])                                      # DAI=-PC1, PAI=+PC2, SAI=-PC3
ENV_METRICS = ["thw_min", "ttc_min", "lead_gap_min", "dist_any_min", "dist_vru_min"]
KIN_REPORT = ["v_avg", "peak_acc", "peak_dec", "long_jerk", "lat_jerk"]
FACTOR_NAMES = ["DAI", "PAI", "SAI"]


# ---------------------------------------------------------------------------
# kinematic + factor pipeline (verbatim reuse of persona_compare_full.py logic)
# ---------------------------------------------------------------------------
def kin(vx, vy):
    vx = np.asarray(vx, float); vy = np.asarray(vy, float); sp = np.hypot(vx, vy)
    ax = np.diff(sp) / DT if len(sp) > 1 else np.array([0.])
    lj = np.diff(ax) / DT if len(ax) > 1 else np.array([0.])
    ay = np.diff(vy) / DT if len(vy) > 1 else np.array([0.])
    aj = np.diff(ay) / DT if len(ay) > 1 else np.array([0.])
    return {"v_avg": float(sp.mean()), "v_std": float(sp.std()),
            "peak_acc": float(ax.max()) if ax.size else 0,
            "peak_dec": float(-ax.min()) if ax.size else 0,
            "long_jerk": float(np.sqrt((lj ** 2).mean())) if lj.size else 0,
            "lat_amax": float(np.abs(ay).max()) if ay.size else 0,
            "lat_jerk": float(np.sqrt((aj ** 2).mean())) if aj.size else 0}


def mvxvy(xy):
    p = np.array([[0., 0.]] + list(xy), float); d = np.diff(p, axis=0) / DT
    return d[:, 0], d[:, 1]


def pctile(val, arr):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return np.searchsorted(arr, val, side="right") / len(arr) - 0.5


def med_env(env_dict, key, metric):
    d = env_dict.get(key, {})
    v = [r[metric] for r in d.values() if isinstance(r, dict) and r.get(metric) is not None]
    return float(np.median(v)) if v else float("nan")


class StyleScorer:
    """Holds the human-reference distributions + PCA loadings and scores dumps."""

    def __init__(self):
        self.ST = json.load(open(ST_PATH))
        self.ENV1 = json.load(open(ENV1_PATH))
        self.LOAD = json.load(open(LOAD_PATH))["loadings"]
        self.L = np.array([self.LOAD[f] for f in ALLF]).T
        # human reference: kin from styletest + env from ENV1["human"]
        Hf = []
        for t, r in self.ST.items():
            if not r.get("vx_ego"):
                continue
            f = kin(r["vx_ego"], r["vy_ego"])
            e = self.ENV1.get("human", {}).get(t, {})
            for lk, sk in zip(ENVK, ENVK_SRC):
                f[lk] = e.get(sk)
            Hf.append(f)
        self.Hf = Hf
        self.HSORT = {}
        for k in ALLF:
            v = np.array([d[k] for d in Hf if d.get(k) is not None], float)
            self.HSORT[k] = np.sort(v)
        HZ = np.array([[pctile(d[k], self.HSORT[k]) for k in ALLF] for d in Hf])
        self.HFAC = (HZ @ self.L.T) * ORI
        self.human_kin = {k: float(np.median([d[k] for d in Hf if d.get(k) is not None])) for k in KIN}

    def factor_rows(self, dump, env_dict, env_key):
        ed = env_dict.get(env_key, {})
        rows = []
        for t, xy in dump.items():
            if t == "_meta":
                continue
            f = kin(*mvxvy(xy))
            e = ed.get(t, {})
            for lk, sk in zip(ENVK, ENVK_SRC):
                f[lk] = e.get(sk)
            z = np.array([pctile(f[k], self.HSORT[k]) for k in ALLF])
            rows.append((z @ self.L.T) * ORI)
        return np.array(rows)

    def factors_pct(self, dump, env_dict, env_key):
        R = self.factor_rows(dump, env_dict, env_key)
        med = np.median(R, axis=0)
        return [float((self.HFAC[:, k] < med[k]).mean() * 100) for k in range(3)]

    def kin_medians(self, dump):
        rows = [kin(*mvxvy(xy)) for t, xy in dump.items() if t != "_meta"]
        return {k: float(np.median([r[k] for r in rows])) for k in KIN}


# ---------------------------------------------------------------------------
# env_interact: compute social features for the sweep dumps (no GPU; needs navsim GT)
# ---------------------------------------------------------------------------
NUPLAN_DEFAULTS = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "/root/workspace/closed_loop/data/navsim/maps",
    "OPENSCENE_DATA_ROOT": "/root/workspace/closed_loop/data/navsim",
    "NAVSIM_DEVKIT_ROOT": os.path.join(AV, "navsim"),
    "NAVSIM_EXP_ROOT": "/mnt/pfs/zhengguantian/autovla/exp",
}


def compute_env_interact(alpha_dumps, out_path, py=sys.executable):
    """Run env_interact.py over the sweep dumps -> social features json at out_path.

    alpha_dumps: {env_key: dump_path}. env_interact also scores 'human' in the
    same pass. Raises on failure (no silent failures)."""
    env = dict(os.environ)
    for k, v in NUPLAN_DEFAULTS.items():
        env.setdefault(k, v)
    env["PYTHONPATH"] = f"{AV}:{AV}/navsim:" + env.get("PYTHONPATH", "")
    env["ONLY_MODELS"] = json.dumps(alpha_dumps)
    env["OUT"] = out_path
    cmd = [py, os.path.join(HERE, "env_interact.py")]
    print(f"[sweep_metrics] computing env_interact for {list(alpha_dumps)} -> {out_path}", flush=True)
    subprocess.run(cmd, env=env, cwd=AV, check=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"env_interact did not produce {out_path}")
    return json.load(open(out_path))


# ---------------------------------------------------------------------------
def parse_alphas_from_dir(outdir):
    alphas = []
    for p in glob.glob(os.path.join(outdir, "persona_a*.json")):
        m = re.match(r"persona_a([0-9.]+)\.json$", os.path.basename(p))
        if m and ".shard" not in os.path.basename(p):
            alphas.append(m.group(1))
    return sorted(set(alphas), key=float)


def fmt(v, w=8, p=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'-':>{w}}"
    return f"{v:>{w}.{p}f}"


def build_table(rows, alphas, teacher_row="DDv2 teacher", l2_key="l2_to_ddv2", l2_col="L2->DDv2"):
    """rows: dict name -> metrics dict. Returns a human-readable multi-section table."""
    lines = []
    order = ["human", "base(AutoVLA)", teacher_row] + [f"alpha={a}" for a in alphas]

    l2w = max(10, len(l2_col) + 2)  # widen for long teacher names; ddv2 ('L2->DDv2') stays 10
    lines.append("=== YDSP factors (percentile vs human; 50=human-like, higher=more aggressive) ===")
    lines.append(f"{'model':<16}{'DAI':>8}{'PAI':>8}{'SAI':>8}{l2_col:>{l2w}}")
    for n in order:
        r = rows.get(n)
        if not r:
            continue
        f = r["ydsp"]
        l2 = r.get(l2_key)
        lines.append(f"{n:<16}{fmt(f[0],8,1)}{fmt(f[1],8,1)}{fmt(f[2],8,1)}{fmt(l2,l2w,2)}")

    lines.append("")
    lines.append("=== kinematic medians (per-scene median trajectory) ===")
    lines.append(f"{'model':<16}{'v_avg':>8}{'peak_acc':>10}{'peak_dec':>10}{'long_jerk':>11}{'lat_jerk':>10}")
    for n in order:
        r = rows.get(n)
        if not r or not r.get("kin"):
            continue
        k = r["kin"]
        lines.append(f"{n:<16}{fmt(k['v_avg'])}{fmt(k['peak_acc'],10)}{fmt(k['peak_dec'],10)}"
                     f"{fmt(k['long_jerk'],11)}{fmt(k['lat_jerk'],10)}")

    lines.append("")
    lines.append("=== social metrics (median over valid scenes) ===")
    lines.append(f"{'model':<16}" + "".join(f"{m.replace('_min',''):>12}" for m in ENV_METRICS))
    for n in order:
        r = rows.get(n)
        if not r or not r.get("social"):
            continue
        s = r["social"]
        lines.append(f"{n:<16}" + "".join(fmt(s.get(m), 12) for m in ENV_METRICS))
    return "\n".join(lines)


def make_plot(rows, alphas, out_png, teacher_row="DDv2 teacher", l2_key="l2_to_ddv2", teacher_disp="DDv2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for fp in ("/mnt/pfs/zhengguantian/NotoSansCJKsc-Regular.otf",
               "/mnt/pfs/zhengguantian/autovla/NotoSansCJKsc-Regular.otf"):
        try:
            fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False

    af = [float(a) for a in alphas]
    base = rows["base(AutoVLA)"]; ddv2 = rows[teacher_row]
    panels = [("DAI", lambda r: r["ydsp"][0], base["ydsp"][0], ddv2["ydsp"][0], 50.0),
              ("PAI", lambda r: r["ydsp"][1], base["ydsp"][1], ddv2["ydsp"][1], 50.0),
              ("SAI", lambda r: r["ydsp"][2], base["ydsp"][2], ddv2["ydsp"][2], 50.0),
              (f"L2 -> {teacher_disp} (YDSP)", (lambda k: (lambda r: r.get(k)))(l2_key),
               base.get(l2_key), 0.0, None)]
    for m in ENV_METRICS:
        panels.append((m, (lambda mm: (lambda r: r["social"].get(mm)))(m),
                       base["social"].get(m), ddv2["social"].get(m),
                       rows["human"]["social"].get(m)))

    n = len(panels); ncol = 3; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, (title, getter, bval, dval, hval) in zip(axes, panels):
        ys = [getter(rows[f"alpha={a}"]) for a in alphas]
        ax.plot(af, ys, "o-", color="#e6194B", lw=2.2, ms=7, label="sweep")
        if bval is not None and not (isinstance(bval, float) and np.isnan(bval)):
            ax.axhline(bval, color="#888", ls="-.", lw=1.4, label="base AutoVLA")
        if dval is not None and not (isinstance(dval, float) and np.isnan(dval)):
            ax.axhline(dval, color="#3cb44b", ls="--", lw=1.8, label=f"{teacher_disp} target")
        if hval is not None and not (isinstance(hval, float) and np.isnan(hval)):
            ax.axhline(hval, color="#999", ls=":", lw=1.2, label="human")
        ax.set_title(title, fontsize=11); ax.set_xlabel("alpha"); ax.grid(alpha=0.3)
        ax.set_xticks(af)
    for ax in axes[len(panels):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Persona sweep: style dose-response vs base / {teacher_disp} / human anchors",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compute & save persona-sweep style metrics.")
    ap.add_argument("--outdir", required=True, help="dir with persona_a*.json (must be on PFS)")
    ap.add_argument("--alphas", default="", help="comma list, e.g. 0.5,0.75,1.0 (default: auto-detect)")
    ap.add_argument("--env-json", default="", help="reuse this precomputed env_interact json (skip recompute)")
    ap.add_argument("--env-key-tmpl", default="a{alpha}",
                    help="env_interact key template per alpha (default 'a{alpha}')")
    ap.add_argument("--no-plot", action="store_true", help="skip style_metrics.png")
    ap.add_argument("--teacher", default="ddv2", choices=sorted(TEACHER_ANCHORS),
                    help="comparison-anchor teacher (default 'ddv2' == diffusiondrivev2; "
                         "byte-identical to legacy behaviour when unset)")
    ap.add_argument("--out-tag", default="",
                    help="suffix for output basenames, e.g. '_ownteacher' -> "
                         "style_metrics_ownteacher.{json,txt,png} (default '' == legacy names)")
    args = ap.parse_args()

    teacher = args.teacher.lower()
    if teacher not in TEACHER_ANCHORS:
        sys.exit(f"[sweep_metrics] unknown --teacher {teacher!r}; choices: {sorted(TEACHER_ANCHORS)}")
    teacher_file, teacher_env_key, teacher_disp = TEACHER_ANCHORS[teacher]
    TEACHER_DUMP = os.path.join(COMPARE_FULL, teacher_file)
    teacher_row = f"{teacher_disp} teacher"
    # l2 field name stays 'l2_to_ddv2' for the default so legacy JSON is byte-identical.
    l2_key = "l2_to_ddv2" if teacher == "ddv2" else f"l2_to_{teacher}"
    l2_col = f"L2->{teacher_disp}"
    if not os.path.exists(TEACHER_DUMP):
        sys.exit(f"[sweep_metrics] teacher anchor dump not found: {TEACHER_DUMP}")

    outdir = os.path.abspath(args.outdir)
    if not outdir.startswith("/mnt/pfs/"):
        print(f"[sweep_metrics] WARNING: outdir {outdir} is not under /mnt/pfs (repo rule: outputs go to PFS)",
              file=sys.stderr)
    if not os.path.isdir(outdir):
        sys.exit(f"[sweep_metrics] outdir not found: {outdir}")

    alphas = [a for a in args.alphas.split(",") if a] or parse_alphas_from_dir(outdir)
    if not alphas:
        sys.exit(f"[sweep_metrics] no persona_a*.json found in {outdir}")
    alphas = sorted(set(alphas), key=float)
    dumps = {a: os.path.join(outdir, f"persona_a{a}.json") for a in alphas}
    for a, p in dumps.items():
        if not os.path.exists(p):
            sys.exit(f"[sweep_metrics] missing dump: {p}")
    print(f"[sweep_metrics] outdir={outdir}  alphas={alphas}", flush=True)

    # --- env_interact social features for the sweep ---
    if args.env_json:
        sweep_env = json.load(open(args.env_json))
        env_src = args.env_json
        env_key = lambda a: args.env_key_tmpl.format(alpha=a)
    else:
        env_out = os.path.join(outdir, "env_interact.json")
        alpha_dumps = {f"a{a}": dumps[a] for a in alphas}
        sweep_env = compute_env_interact(alpha_dumps, env_out)
        env_src = env_out
        env_key = lambda a: f"a{a}"

    scorer = StyleScorer()
    base_dump = json.load(open(BASE_DUMP))
    teacher_dump = json.load(open(TEACHER_DUMP))

    rows = {}
    # anchors
    base_ydsp = scorer.factors_pct(base_dump, scorer.ENV1, "autovla")
    teacher_ydsp = scorer.factors_pct(teacher_dump, scorer.ENV1, teacher_env_key)
    dd = np.array(teacher_ydsp)
    rows["base(AutoVLA)"] = {
        "ydsp": base_ydsp, "kin": scorer.kin_medians(base_dump),
        "social": {m: med_env(scorer.ENV1, "autovla", m) for m in ENV_METRICS},
        l2_key: float(np.linalg.norm(np.array(base_ydsp) - dd))}
    rows[teacher_row] = {
        "ydsp": teacher_ydsp, "kin": scorer.kin_medians(teacher_dump),
        "social": {m: med_env(scorer.ENV1, teacher_env_key, m) for m in ENV_METRICS},
        l2_key: 0.0}
    rows["human"] = {
        "ydsp": [50.0, 50.0, 50.0], "kin": scorer.human_kin,
        "social": {m: med_env(scorer.ENV1, "human", m) for m in ENV_METRICS},
        l2_key: float(np.linalg.norm(np.array([50.0, 50.0, 50.0]) - dd))}

    # per-alpha
    for a in alphas:
        dump = json.load(open(dumps[a]))
        ek = env_key(a)
        ydsp = scorer.factors_pct(dump, sweep_env, ek)
        rows[f"alpha={a}"] = {
            "ydsp": ydsp, "kin": scorer.kin_medians(dump),
            "social": {m: med_env(sweep_env, ek, m) for m in ENV_METRICS},
            l2_key: float(np.linalg.norm(np.array(ydsp) - dd)),
            "env_key": ek, "n_tokens": len([k for k in dump if k != "_meta"])}

    # --- persist ---
    table = build_table(rows, alphas, teacher_row=teacher_row, l2_key=l2_key, l2_col=l2_col)
    anchors = {"base": BASE_DUMP, "ddv2": DDV2_DUMP, "anchor_env": ENV1_PATH,
               "pca_loadings": LOAD_PATH, "styletest": ST_PATH}
    if teacher != "ddv2":
        anchors["teacher"] = TEACHER_DUMP
        anchors["teacher_name"] = teacher
        anchors["teacher_env_key"] = teacher_env_key
    structured = {
        "outdir": outdir, "alphas": alphas,
        "anchors": anchors,
        "env_source": env_src,
        "factor_names": FACTOR_NAMES, "kin_keys": KIN, "social_keys": ENV_METRICS,
        "rows": rows,
    }
    tag = args.out_tag
    json_path = os.path.join(outdir, f"style_metrics{tag}.json")
    txt_path = os.path.join(outdir, f"style_metrics{tag}.txt")
    json.dump(structured, open(json_path, "w"), indent=2)
    with open(txt_path, "w") as fh:
        fh.write(table + "\n")
    print("\n" + table)
    print(f"\n[sweep_metrics] saved: {json_path}")
    print(f"[sweep_metrics] saved: {txt_path}")

    png_path = os.path.join(outdir, f"style_metrics{tag}.png")
    if not args.no_plot:
        try:
            make_plot(rows, alphas, png_path, teacher_row=teacher_row,
                      l2_key=l2_key, teacher_disp=teacher_disp)
            print(f"[sweep_metrics] saved: {png_path}")
        except Exception as e:
            # plot is optional; log loudly but don't fail the metrics run
            print(f"[sweep_metrics] WARNING: plot failed ({e!r}); metrics json/txt still saved",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
