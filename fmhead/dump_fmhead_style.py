"""dump_fmhead_style.py -- dump FMHead-decoded ego trajectories per token, in the SAME
`{token: [[x,y],...]}` format the youdrive style pipeline (sweep_metrics.py / StyleScorer /
env_interact) consumes. Serves BOTH:

  * DELIVERABLE 1 (style measurement): one dump at alpha=0 (style OFF, s=0) -> compare
    FMHead(s=0) vs codebook AutoVLA (compare_full/autovla.json) vs human via sweep_metrics.
  * DELIVERABLE 2 (controllability): a persona LoRA attached + ALPHAS sweep of the HEAD's
    style gain -> persona_a{alpha}.json dumps -> sweep_metrics gives the dose-response
    (DAI/PAI/SAI + kin + social) toward the DDv2 teacher, and gap-closure 1 - D(1)/D(0).

Style axis here is the FMHead's s-vector gain (NOT the LoRA scaling): with the persona LoRA
attached, s = alpha*(styled_anchor - base_anchor); CONTENT (cross-attn ctx) always comes
from the BASE VLM so scene/safety is decoupled from style. alpha=0 -> s=0 -> pure content.

Selection modes:
  * medoid (default) / mean: scene-free (closest-to-mean-endpoint pick). Isolates the style
    axis from PDM re-ranking -> the right input for the alpha controllability dose-response.
  * pdm: the DEPLOYED selection (the one that scored 0.9088). Decode N candidates, score each
    with PDM_Reward keyed by the token (== SceneMetadata.initial_token; styletest tokens are a
    subset of navtest so METRIC_CACHE=metric_cache_navtest_v1 covers them), argmax. Scene-free
    (uses the token directly). SLOW: N x #tokens PDM cache reads -> shard with NUM_SHARDS/SHARD_IDX.

Env:
  CONFIG        (default config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml)
  BASE_CKPT     base VLM ckpt (default AutoVLA_PDMS_89.ckpt; for full-FT pass full_ft_base.ckpt)
  FMHEAD_CKPT   trained FMHead decoder .pt / fm_decoder.pt        [required]
  FMHEAD_NORM   normalizer json (default traj_norm_stats_gt.json; overridden by baked-in buffers)
  PERSONA_ADAPTERS  persona LoRA dir (set -> style ON). unset -> style OFF (s=0, D1).
  ALPHAS        csv head style gains (default "0" for D1; e.g. "0,0.25,0.5,0.75,1.0" for D2)
  CFG_WEIGHT    classifier-free-guidance weight on style (default 1.0; >1 sharpens toward style)
  FMHEAD_N      num samples (default 16)   FMHEAD_STEPS  euler steps (default 30)
  SELECT        medoid (default) | mean
  TOKENS        json list of tokens (default: all styletest tokens present in JSON_DIR)
  HIONLY=1      restrict to high-elasticity scenario types
  OUTDIR        (default /mnt/pfs/zhengguantian/autovla/persona/fmhead_style)
  NUM_SHARDS/SHARD_IDX  token sharding for multi-GPU
"""
import os, json
import numpy as np, torch, yaml
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD = "/root/workspace/fmhead"
CONFIG = os.environ.get("CONFIG", os.path.join(REPO, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
BASE_CKPT = os.environ.get("BASE_CKPT", "/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt")
FMHEAD_CKPT = os.environ["FMHEAD_CKPT"]
FMHEAD_NORM = os.environ.get("FMHEAD_NORM", os.path.join(FMHEAD, "traj_norm_stats_gt.json"))
JSON_DIR = os.path.join(REPO, "dataset/nuplan/navtest_nocot")
OUTDIR = os.environ.get("OUTDIR", "/mnt/pfs/zhengguantian/autovla/persona/fmhead_style")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0").split(",")]
CFG_WEIGHT = float(os.environ.get("CFG_WEIGHT", "1.0"))
NSAMP = int(os.environ.get("FMHEAD_N", "16"))
NSTEPS = int(os.environ.get("FMHEAD_STEPS", "30"))
SELECT = os.environ.get("SELECT", "medoid")
# metric cache for SELECT=pdm. styletest tokens are a subset of navtest, so the navtest
# cache covers them (verified: 4049/4049) -> pdm-select on styletest tokens is apples-to-apples
# with the styletest anchors (compare_full/autovla.json etc).
METRIC_CACHE = os.environ.get("METRIC_CACHE", "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
ST = json.load(open(os.environ.get("STYLETEST", "/root/workspace/tools/scorer/data/styledrive/styletest.json")))


def _parse_constraint(s):
    """FMHEAD_CONSTRAINT env -> fm3 constraint dict for the FMHead agent.

    ADDITIVE: unset/empty -> None -> byte-identical fm2 xy decode (existing style path
    unchanged). Accepts JSON, or the compact Hydra dict form the PDMS harness already
    passes, e.g. '{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
    (plain yaml/OmegaConf mis-parse this space-less flow map, so parse it here). Numeric
    values are coerced to float; `mode` stays a string."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    out = {}
    for part in s.strip().lstrip("{").rstrip("}").split(","):
        if not part.strip():
            continue
        k, _, v = part.partition(":")
        k, v = k.strip(), v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


CONSTRAINT = _parse_constraint(os.environ.get("FMHEAD_CONSTRAINT"))


def pick_tokens():
    t = os.environ.get("TOKENS")
    if t:
        toks = json.load(open(t))
    elif os.environ.get("HIONLY"):
        HI = {"side_ego_to_main", "carpark_areas", "lane_change", "crosswalks", "unprotected_intersections"}
        toks = [k for k, r in ST.items() if r.get("scenario_type") in HI
                and os.path.exists(os.path.join(JSON_DIR, k + ".json"))]
    else:
        toks = [k for k in ST if os.path.exists(os.path.join(JSON_DIR, k + ".json"))]
    # LIMIT (ADDITIVE): cap the token set to the first N (deterministic order) for a fast
    # subset sweep on 1 GPU. Unset/0 -> all tokens (unchanged). Applied BEFORE sharding so
    # a single-GPU LIMIT run and a sharded full run both start from the same ordered list.
    lim = int(os.environ.get("LIMIT", "0"))
    if lim > 0:
        toks = toks[:lim]
    ns = int(os.environ.get("NUM_SHARDS", "1")); si = int(os.environ.get("SHARD_IDX", "0"))
    return toks[si::ns] if ns > 1 else toks


def main():
    os.chdir(REPO); os.makedirs(OUTDIR, exist_ok=True)
    import sys
    sys.path.insert(0, FMHEAD)
    from fmhead_navsim_agent import FMHeadAutoVLAAgent

    tokens = pick_tokens()
    persona = os.environ.get("PERSONA_ADAPTERS")
    style_off = not persona
    # use_style_adapter: explicit env OR implied when a persona LoRA is attached with any alpha>0
    # (i.e. we intend to inject style -> the ckpt is a style-adapter ckpt that must be built WITH
    # the adapter submodule so its weights load strictly). The agent ALSO auto-detects adapter
    # keys in the ckpt as a safety net.
    _env_flag = os.environ.get("FMHEAD_USE_STYLE_ADAPTER", os.environ.get("USE_STYLE_ADAPTER", "0"))
    # z-encoder ckpts have NO adapter keys -> when FMHEAD_USE_Z_ENCODER is set we must NOT force
    # the adapter (else the plain+z head would fail the strict load). The agent auto-detects the
    # z_encoder from the ckpt keys regardless; this flag only suppresses the adapter heuristic.
    _zenc_flag = os.environ.get("FMHEAD_USE_Z_ENCODER", "0").lower() in ("1", "true", "yes")
    _content_source = os.environ.get("FMHEAD_CONTENT_SOURCE", "base")
    # content_source=interp is the inference-only LATENT CONTENT INTERPOLATION mode: it forces the
    # AdaLN style s=0 and decodes with a PLAIN head (fm3-kin ep6 head has NO adapter/z keys), so we
    # must NOT force the style-adapter heuristic (it would try to build/strict-load adapter weights
    # that do not exist and hard-fail). z-encoder is likewise irrelevant here.
    _interp = (_content_source == "interp")
    use_style_adapter = (not _zenc_flag) and (not _interp) and (
        (_env_flag.lower() in ("1", "true", "yes")) or
        (bool(persona) and any(float(a) > 0 for a in ALPHAS)))
    print(f"[fmstyle] {len(tokens)} tokens | style_off={style_off} persona={persona} "
          f"use_style_adapter={use_style_adapter} z_encoder={_zenc_flag} content_source={_content_source} "
          f"alphas={ALPHAS} cfg_w={CFG_WEIGHT} N={NSAMP} steps={NSTEPS} select={SELECT} "
          f"constraint={CONSTRAINT}", flush=True)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    ts = TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                            interval_length=cfg["model"]["trajectory"]["interval_length"])
    agent = FMHeadAutoVLAAgent(
        trajectory_sampling=ts, checkpoint_path=BASE_CKPT, sensor_data_path=".",
        config_path=CONFIG, lora_conf={"use_lora": False}, device="cuda",
        fmhead_ckpt_path=FMHEAD_CKPT, fmhead_normalizer_path=FMHEAD_NORM,
        fmhead_style_off=style_off, fmhead_style_alpha=1.0,
        fmhead_num_samples=NSAMP, fmhead_num_steps=NSTEPS, fmhead_cfg_weight=CFG_WEIGHT,
        fmhead_select_mode=("medoid" if SELECT == "mean" else SELECT),
        fmhead_metric_cache_path=(METRIC_CACHE if SELECT == "pdm" else None),
        fmhead_use_style_adapter=use_style_adapter,
        fmhead_content_source=_content_source,
        fmhead_constraint=CONSTRAINT)         # fm3: None -> unchanged fm2 xy decode
    agent.initialize()  # loads base (+ PERSONA_ADAPTERS LoRA if set), FMHead ckpt, (+ PDM_Reward if pdm)
    agent.autovla.eval()

    from navsim.common.dataclasses import Trajectory
    np_poses = agent._trajectory_sampling.num_poses

    def decode_token(ai, tok):
        """Return (num_poses,2) xy for one token, per SELECT. pdm-select is scene-FREE: the
        token IS the metric-cache key (== SceneMetadata.initial_token), so we score the N
        candidates with the agent's PDM_Reward and argmax -- the DEPLOYED (0.9088) selection."""
        if SELECT == "pdm":
            features = agent._build_features(ai)
            ctx, ctx_mask, s = agent._extract_ctx_s(features)
            # v0 (ego initial speed from vehicle_velocity) MUST be forwarded to the kinematic
            # decode -- identical to the medoid/fm_predict path -- else fm3-kin integrates the
            # unicycle from v0=0 and the (a_long,yaw_rate) samples produce wrong speeds. Returns
            # None in xy (fm2, constraint=None) mode -> decode_all ignores it (path unchanged).
            v0 = agent._v0_from_features(features)
            cands = agent._fm_decoder.decode_all(ctx, s, ctx_mask=ctx_mask, v0=v0)[0].float().cpu().numpy()  # (N,T,3)
            scores = []
            for k in range(cands.shape[0]):
                try:
                    tr = Trajectory(cands[k][:np_poses], agent._trajectory_sampling)
                    scores.append(float(agent._pdm.rl_pdm_score(tr, tok)))
                except Exception as e:  # noqa: BLE001 - a failed candidate must not win
                    scores.append(-1.0)
            return cands[int(np.argmax(scores))][:np_poses, :2]
        traj, _ = agent.compute_trajectory(ai)                  # medoid|mean: scene-free
        return np.asarray(traj.poses)[:np_poses, :2]

    ns = int(os.environ.get("NUM_SHARDS", "1")); si = int(os.environ.get("SHARD_IDX", "0"))
    suffix = f".shard{si}" if ns > 1 else ""
    for a in ALPHAS:
        agent._fm_alpha = float(a)                     # HEAD style gain (dynamic)
        agent._fm_decoder.cfg_weight = CFG_WEIGHT
        res, fail = {}, []
        for tok in tokens:
            try:
                ai = json.load(open(os.path.join(JSON_DIR, tok + ".json")))
                with torch.no_grad():
                    xy = decode_token(ai, tok)                  # SELECT: medoid|mean|pdm
                res[tok] = np.asarray(xy).astype(float).tolist()
            except Exception as e:  # noqa: BLE001 - report, never silently drop
                fail.append(tok)
                if len(fail) <= 3:
                    print(f"  a={a} fail {tok}: {e!r}"[:160], flush=True)
        meta = {"model": f"FMHead{'+DDv2' if persona else '(s=0)'}@a{a}",
                "horizon_s": float(ts.time_horizon), "alpha": float(a), "cfg_weight": CFG_WEIGHT,
                "n_samples": NSAMP, "steps": NSTEPS, "select": SELECT, "style_off": style_off,
                "n_pts": len(next(iter(res.values()))) if res else 0,
                "frame": "ego BEV, x=forward(m), y=left(m), cumulative positions", "n_fail": len(fail)}
        out = {"_meta": meta}; out.update(res)
        p = os.path.join(OUTDIR, f"persona_a{a:g}{suffix}.json")  # :g -> 0.0->'0', 1.0->'1' (match sweep_metrics naming)
        json.dump(out, open(p, "w"))
        print(f"[fmstyle] alpha={a}: {len(res)}/{len(tokens)} ok ({len(fail)} fail) -> {p}", flush=True)


if __name__ == "__main__":
    main()
