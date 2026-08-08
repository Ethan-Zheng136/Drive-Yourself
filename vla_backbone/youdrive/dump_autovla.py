"""Dump AutoVLA planned ego trajectories for a fixed set of navtest tokens.

Builds the AutoVLAAgent directly from the training config + checkpoint and
calls compute_trajectory() per token, then writes cumulative ego-frame (x, y)
positions to a unified JSON.
"""
import os
import json
import traceback
import yaml

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.autovla_agent import AutoVLAAgent

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
CKPT = "/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt"
CFG = os.path.join(REPO, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml")
JSON_DIR = os.path.join(REPO, "dataset/nuplan/navtest_nocot")
TOKENS_FILE = os.environ.get("TOKENS_FILE", "/mnt/pfs/zhengguantian/autovla/compare/tokens_anc.json")
OUT_FILE = os.environ.get("OUT_FILE", "/mnt/pfs/zhengguantian/autovla/compare_anc/autovla.json")
SHARD_IDX = int(os.environ.get("SHARD_IDX") or "0")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS") or "1")


def main():
    os.chdir(REPO)

    with open(TOKENS_FILE, "r") as f:
        tokens = json.load(f)

    if NUM_SHARDS > 1:
        tokens = tokens[SHARD_IDX::NUM_SHARDS]
        print(f"shard {SHARD_IDX}/{NUM_SHARDS}: {len(tokens)} tokens", flush=True)

    with open(CFG, "r") as f:
        cfg = yaml.safe_load(f)
    interval_length = cfg["model"]["trajectory"]["interval_length"]
    time_horizon = cfg["model"]["trajectory"]["time_horizon"]

    trajectory_sampling = TrajectorySampling(
        time_horizon=time_horizon,
        interval_length=interval_length,
    )

    lora_conf = {
        "use_lora": False,
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "bias": "none",
    }

    agent = AutoVLAAgent(
        trajectory_sampling=trajectory_sampling,
        checkpoint_path=CKPT,
        sensor_data_path=".",
        config_path=CFG,
        lora_conf=lora_conf,
        device="cuda",
    )

    # Equivalent to agent.initialize(), but load the checkpoint onto CPU first.
    # Deserializing this checkpoint directly to the MIG cuda device trips an
    # NVML INTERNAL ASSERT in the caching allocator; copying CPU tensors into
    # the already-on-GPU model via load_state_dict avoids that path.
    assert not agent.lora_conf.get("use_lora", False), "lora path not handled here"
    state_dict = torch.load(CKPT, map_location="cpu")["state_dict"]
    agent.autovla.load_state_dict(
        {k.replace("autovla.", ""): v for k, v in state_dict.items()}, strict=False
    )

    results = {}
    failed = []
    n_pts = None

    for i, token in enumerate(tokens):
        print(f"[{i + 1}/{len(tokens)}] token={token}", flush=True)
        try:
            json_path = os.path.join(JSON_DIR, f"{token}.json")
            with open(json_path, "r") as f:
                agent_input = json.load(f)

            with torch.no_grad():
                trajectory, _cot = agent.compute_trajectory(agent_input)

            poses = np.asarray(trajectory.poses)  # [N, 3] -> (x, y, heading), ego frame
            xy = poses[:, :2].astype(float).tolist()
            results[token] = xy
            if n_pts is None:
                n_pts = len(xy)
            print(f"    ok: {len(xy)} pts", flush=True)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed.append(token)

    out = {
        "_meta": {
            "model": "AutoVLA",
            "horizon_s": float(time_horizon),
            "n_pts": int(n_pts) if n_pts is not None else 0,
            "frame": "ego BEV, x=forward(m), y=left(m), cumulative positions",
        }
    }
    if failed:
        out["_meta"]["failed"] = failed
    out.update(results)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {OUT_FILE}: {len(results)}/{len(tokens)} succeeded, "
          f"n_pts={n_pts}, horizon_s={time_horizon}", flush=True)


if __name__ == "__main__":
    main()
