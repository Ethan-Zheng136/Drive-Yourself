"""Dose-response: load AutoVLA base + a trained LoRA adapter, sweep alpha (LoRA strength),
dump ego trajectories per alpha. alpha=0 -> base, alpha=1 -> full LoRA, alpha>1 -> extrapolate.
Then ydsp.py / style_metrics on each dump shows whether the persona shifts DAI/PAI/SAI monotonically.

Env:
  CONFIG   (default config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml)
  BASE_CKPT(default AutoVLA_PDMS_89.ckpt)
  ADAPTER  (peft adapter dir, e.g. .../lora_ckpts/<date>/lora_step1000)  [required]
  ALPHAS   (csv, default "0,0.5,1.0,1.5")
  TOKENS   (json list; default = high-elasticity subset built here)
  OUTDIR   (default /mnt/pfs/zhengguantian/autovla/compare_persona)
"""
import os, json, traceback
import numpy as np, torch, yaml
from peft import PeftModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.autovla_agent import AutoVLAAgent

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
CONFIG = os.environ.get("CONFIG", os.path.join(REPO, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
BASE_CKPT = os.environ.get("BASE_CKPT", "/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt")
ADAPTER = os.environ["ADAPTER"]
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0,0.5,1.0,1.5").split(",")]
JSON_DIR = os.path.join(REPO, "dataset/nuplan/navtest_nocot")
OUTDIR = os.environ.get("OUTDIR", "/mnt/pfs/zhengguantian/autovla/compare_persona")
ST = json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))

def pick_tokens():
    t = os.environ.get("TOKENS")
    if t:
        toks = json.load(open(t))
    elif os.environ.get("HIONLY"):
        HI = {"side_ego_to_main","carpark_areas","lane_change","crosswalks","unprotected_intersections"}
        toks = [k for k, r in ST.items() if r.get("scenario_type") in HI
                and os.path.exists(os.path.join(JSON_DIR, k + ".json"))]
    else:  # default: ALL styletest tokens (full) -> directly comparable to the 8-model YDSP
        toks = [k for k in ST if os.path.exists(os.path.join(JSON_DIR, k + ".json"))]
    ns = int(os.environ.get("NUM_SHARDS", "1")); si = int(os.environ.get("SHARD_IDX", "0"))
    if ns > 1:
        toks = toks[si::ns]
    return toks

def main():
    os.chdir(REPO); os.makedirs(OUTDIR, exist_ok=True)
    tokens = pick_tokens()
    print(f"[sweep] {len(tokens)} eval tokens, alphas={ALPHAS}, adapter={ADAPTER}", flush=True)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    ts = TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                            interval_length=cfg["model"]["trajectory"]["interval_length"])
    agent = AutoVLAAgent(trajectory_sampling=ts, checkpoint_path=BASE_CKPT, sensor_data_path=".",
                         config_path=CONFIG, lora_conf={"use_lora": False}, device="cuda")
    # load base weights
    sd = torch.load(BASE_CKPT, map_location="cpu")["state_dict"]
    agent.autovla.load_state_dict({k.replace("autovla.", ""): v for k, v in sd.items()}, strict=False)
    # attach trained LoRA adapter(s). ADAPTER may be a comma-separated list for a
    # MULTI-PERSONA convex mix; WEIGHTS (comma list, same length) sets each persona's
    # weight. Single adapter -> WEIGHTS defaults to 1 (original behavior). At a given
    # alpha the effective LoRA delta = alpha * sum_i(weight_i * delta_i).
    _paths = [p for p in ADAPTER.split(",") if p]
    _weights = [float(w) for w in os.environ.get("WEIGHTS", "1").split(",")]
    assert len(_weights) == len(_paths), f"WEIGHTS({len(_weights)}) must match ADAPTER({len(_paths)})"
    agent.autovla.vlm = PeftModel.from_pretrained(agent.autovla.vlm, _paths[0], adapter_name="a0").to("cuda")
    for _i, _p in enumerate(_paths[1:], 1):
        agent.autovla.vlm.load_adapter(_p, adapter_name=f"a{_i}")
    _names = [f"a{_i}" for _i in range(len(_paths))]
    _name2w = dict(zip(_names, _weights))
    # PeftModel.set_adapter rejects a list; base_model.set_adapter activates ALL (additive).
    agent.autovla.vlm.base_model.set_adapter(_names)
    agent.autovla.eval()
    print(f"[sweep] personas={_paths} weights={_weights}", flush=True)

    # record each LoRA layer's base scaling so we can do scaling = base*weight*alpha
    base_scaling = {}
    for name, mod in agent.autovla.vlm.named_modules():
        if hasattr(mod, "scaling") and isinstance(getattr(mod, "scaling"), dict):
            base_scaling[name] = dict(mod.scaling)

    def set_alpha(a):
        for name, mod in agent.autovla.vlm.named_modules():
            if name in base_scaling:
                for adp, s in base_scaling[name].items():
                    mod.scaling[adp] = s * _name2w.get(adp, 1.0) * a

    for a in ALPHAS:
        set_alpha(a)
        res, fail = {}, []
        for i, tok in enumerate(tokens):
            try:
                ai = json.load(open(os.path.join(JSON_DIR, tok + ".json")))
                with torch.no_grad():
                    traj, _ = agent.compute_trajectory(ai)
                res[tok] = np.asarray(traj.poses)[:, :2].astype(float).tolist()
            except Exception as e:
                fail.append(tok)
                if len(fail) <= 3: print(f"  a={a} fail {tok}: {e!r}"[:120], flush=True)
        out = {"_meta": {"model": f"AutoVLA+vDDv2@a{a}", "horizon_s": float(ts.time_horizon),
                         "n_pts": len(next(iter(res.values()))) if res else 0, "alpha": a,
                         "frame": "ego BEV, x=forward(m), y=left(m), cumulative positions"}}
        out.update(res)
        ns = int(os.environ.get("NUM_SHARDS", "1")); si = int(os.environ.get("SHARD_IDX", "0"))
        suffix = f".shard{si}" if ns > 1 else ""
        p = os.path.join(OUTDIR, f"persona_a{a}{suffix}.json")
        json.dump(out, open(p, "w"))
        print(f"[sweep] alpha={a}: {len(res)}/{len(tokens)} ok -> {p}", flush=True)

if __name__ == "__main__":
    main()
