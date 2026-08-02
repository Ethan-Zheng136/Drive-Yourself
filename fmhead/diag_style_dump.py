"""diag_style_dump.py -- efficient style diagnostic dump (Steps 2 & 3).

Loads the DDv2 style-adapter head + full-FT base VLM + persona LoRA ONCE. For each
styletest token it does the (expensive) VLM feature extraction ONCE (2 forwards:
base + styled) to get ctx, ctx_mask, and the full style delta d = (styled - base).
Because s = alpha * d is LINEAR in alpha, the whole alpha sweep is then just cheap
FMHead decodes (no extra VLM forwards).

Outputs (dump format {token: [[x,y],...]} matching sweep_metrics.py):
  * MEDOID dir: persona_a{0,0.25,0.5,0.75,1}.json  (SELECT=medoid dose-response, Step 2)
  * PDM    dir: persona_a1.json                     (SELECT=pdm at alpha=1,   Step 3)

WASHOUT ISOLATION: at alpha=1 the SAME candidate pool (one decode_all) is scored by
BOTH the medoid rule and the PDM rule, so medoid@1 vs pdm@1 differ ONLY in selection.
Per-token seeding makes the alpha sweep share noise across alphas (isolates the alpha
axis from sampling noise).

Read-only w.r.t. all checkpoints. Nothing is written outside the two PFS OUTDIRs.
"""
import os, sys, json, time
import numpy as np, torch, yaml
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD = "/root/workspace/fmhead"
sys.path.insert(0, FMHEAD)

CONFIG = os.path.join(REPO, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml")
BASE_CKPT = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/2026-07-14_15-49-24/full_ft_base.ckpt"
FMHEAD_CKPT = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts_style_ddv2/2026-07-16_10-56-07_x4/fmhead_final.pt"
FMHEAD_NORM = os.path.join(FMHEAD, "traj_norm_stats_gt.json")
PERSONA = "/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/ddv2_big_final/lora_final"
JSON_DIR = os.path.join(REPO, "dataset/nuplan/navtest_nocot")
STYLETEST = "/root/workspace/tools/scorer/data/styledrive/styletest.json"
METRIC_CACHE = "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1"

OUTDIR_MEDOID = os.environ.get("OUTDIR_MEDOID", "/mnt/pfs/zhengguantian/autovla/persona/fmhead_style_medoid_diag")
OUTDIR_PDM = os.environ.get("OUTDIR_PDM", "/mnt/pfs/zhengguantian/autovla/persona/fmhead_style_pdm1_diag")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0,0.25,0.5,0.75,1.0").split(",")]
NSAMP = int(os.environ.get("FMHEAD_N", "16"))
NSTEPS = int(os.environ.get("FMHEAD_STEPS", "30"))
CFG_WEIGHT = float(os.environ.get("CFG_WEIGHT", "1.0"))
LIMIT = int(os.environ.get("LIMIT", "0"))   # 0 = all tokens


def tok_seed(tok):
    return abs(hash(tok)) % (2 ** 31)


def main():
    os.chdir(REPO)
    os.makedirs(OUTDIR_MEDOID, exist_ok=True)
    os.makedirs(OUTDIR_PDM, exist_ok=True)
    from fmhead_navsim_agent import FMHeadAutoVLAAgent
    from navsim.common.dataclasses import Trajectory

    ST = json.load(open(STYLETEST))
    tokens = [k for k in ST if os.path.exists(os.path.join(JSON_DIR, k + ".json"))]
    if LIMIT:
        tokens = tokens[:LIMIT]
    print(f"[diag] {len(tokens)} styletest tokens | alphas={ALPHAS} N={NSAMP} steps={NSTEPS} "
          f"cfg_w={CFG_WEIGHT}", flush=True)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    ts = TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                            interval_length=cfg["model"]["trajectory"]["interval_length"])
    # build with pdm select so agent._pdm (PDM_Reward) is loaded; style ON, alpha=1 so
    # _extract_ctx_s returns the FULL delta d = (styled - base).
    agent = FMHeadAutoVLAAgent(
        trajectory_sampling=ts, checkpoint_path=BASE_CKPT, sensor_data_path=".",
        config_path=CONFIG, lora_conf={"use_lora": False}, device="cuda",
        fmhead_ckpt_path=FMHEAD_CKPT, fmhead_normalizer_path=FMHEAD_NORM,
        fmhead_style_off=False, fmhead_style_alpha=1.0,
        fmhead_num_samples=NSAMP, fmhead_num_steps=NSTEPS, fmhead_cfg_weight=CFG_WEIGHT,
        fmhead_select_mode="pdm", fmhead_metric_cache_path=METRIC_CACHE,
        fmhead_use_style_adapter=True)
    os.environ["PERSONA_ADAPTERS"] = PERSONA   # read by the base AutoVLAAgent.initialize()
    agent.initialize()
    agent.autovla.eval()
    dec = agent._fm_decoder
    dec.cfg_weight = CFG_WEIGHT
    np_poses = ts.num_poses

    def medoid_pick(cands_xy):
        # cands_xy: (N, T, 2) -> closest-to-mean-endpoint (same rule as FMHead.select_by_score)
        ep = cands_xy[:, -1, :]
        mean_ep = ep.mean(axis=0, keepdims=True)
        d = np.linalg.norm(ep - mean_ep, axis=-1)
        return cands_xy[int(np.argmin(d))]

    res_medoid = {a: {} for a in ALPHAS}
    res_pdm1 = {}
    fails = []
    t0 = time.time()
    for i, tok in enumerate(tokens):
        try:
            ai = json.load(open(os.path.join(JSON_DIR, tok + ".json")))
            features = agent._build_features(ai)
            with torch.no_grad():
                ctx, ctx_mask, d = agent._extract_ctx_s(features)   # d = 1.0*(styled-base)
                for a in ALPHAS:
                    s = (float(a)) * d
                    torch.manual_seed(tok_seed(tok))                # share noise across alphas
                    cands = dec.decode_all(ctx, s, ctx_mask=ctx_mask)[0].float().cpu().numpy()  # (N,T,3)
                    res_medoid[a][tok] = medoid_pick(cands[:, :np_poses, :2]).astype(float).tolist()
                    if abs(a - 1.0) < 1e-9:
                        # PDM selection on the SAME candidate pool (washout isolation)
                        scores = []
                        for k in range(cands.shape[0]):
                            try:
                                tr = Trajectory(cands[k][:np_poses], ts)
                                scores.append(float(agent._pdm.rl_pdm_score(tr, tok)))
                            except Exception as e:  # noqa: BLE001 - a failed cand must not win
                                scores.append(-1.0)
                        res_pdm1[tok] = cands[int(np.argmax(scores))][:np_poses, :2].astype(float).tolist()
        except Exception as e:  # noqa: BLE001 - report, never silently drop
            fails.append(tok)
            if len(fails) <= 5:
                print(f"  fail {tok}: {e!r}"[:200], flush=True)
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"[diag] {i+1}/{len(tokens)} done ({dt:.0f}s, {dt/(i+1):.2f}s/tok, {len(fails)} fail)", flush=True)

    def meta(alpha, select):
        return {"model": f"FMHead+DDv2@a{alpha}", "select": select, "alpha": float(alpha),
                "cfg_weight": CFG_WEIGHT, "n_samples": NSAMP, "steps": NSTEPS,
                "n_tokens": None, "n_fail": len(fails),
                "frame": "ego BEV, x=forward(m), y=left(m), cumulative positions"}

    for a in ALPHAS:
        out = {"_meta": meta(a, "medoid")}
        out["_meta"]["n_tokens"] = len(res_medoid[a])
        out.update(res_medoid[a])
        p = os.path.join(OUTDIR_MEDOID, f"persona_a{a:g}.json")
        json.dump(out, open(p, "w"))
        print(f"[diag] MEDOID alpha={a}: {len(res_medoid[a])} tok -> {p}", flush=True)

    out = {"_meta": meta(1.0, "pdm")}
    out["_meta"]["n_tokens"] = len(res_pdm1)
    out.update(res_pdm1)
    p = os.path.join(OUTDIR_PDM, "persona_a1.json")
    json.dump(out, open(p, "w"))
    print(f"[diag] PDM alpha=1: {len(res_pdm1)} tok -> {p}", flush=True)
    print(f"[diag] DONE total={time.time()-t0:.0f}s fails={len(fails)}", flush=True)


if __name__ == "__main__":
    main()
