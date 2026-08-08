import json
import os
from pathlib import Path

import torch
import timm

# The transfuser backbone calls timm.create_model(..., pretrained=True) which tries to
# fetch ImageNet weights from HF (fails behind the flaky proxy). The full transfuser
# checkpoint overwrites every backbone weight anyway, so force pretrained=False.
_orig_create_model = timm.create_model


def _no_pretrained_create_model(*args, **kwargs):
    kwargs["pretrained"] = False
    return _orig_create_model(*args, **kwargs)


timm.create_model = _no_pretrained_create_model

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

TOKENS_JSON = os.environ.get("TOKENS_JSON", "/mnt/pfs/zhengguantian/autovla/compare/tokens.json")
OUT_JSON = os.environ.get("OUT_JSON", "/mnt/pfs/zhengguantian/autovla/compare/transfuser.json")
CKPT = "/mnt/pfs/zhengguantian/transfuser/ckpt/transfuser_seed_0.ckpt"

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
# SPLIT=trainval to dump the navtrain12k tokens (they live in the trainval split), default test.
SPLIT = os.environ.get("SPLIT", "test")
LOGS_PATH = DATA_ROOT / "navsim_logs" / SPLIT
SENSOR_PATH = DATA_ROOT / "sensor_blobs" / SPLIT


def main():
    # MERGE mode: combine OUT_JSON.shard* -> OUT_JSON and exit (no model load).
    if os.environ.get("MERGE") == "1":
        import glob
        merged, meta = {}, None
        for fp in sorted(glob.glob(OUT_JSON + ".shard*")):
            d = json.load(open(fp)); meta = meta or d.get("_meta")
            for k, v in d.items():
                if k != "_meta":
                    merged[k] = v
        out = {"_meta": meta} if meta else {}
        out.update(merged)
        json.dump(out, open(OUT_JSON, "w"))
        print(f"[merge] {len(merged)} tokens -> {OUT_JSON}")
        return

    with open(TOKENS_JSON) as f:
        tokens = json.load(f)
    # 8-GPU sharding: each process handles tokens[SHARD_IDX::NUM_SHARDS], writes OUT_JSON.shard{idx}.
    _ns = int(os.environ.get("NUM_SHARDS", "1"))
    _si = int(os.environ.get("SHARD_IDX", "0"))
    out_path = OUT_JSON
    if _ns > 1:
        tokens = tokens[_si::_ns]
        out_path = f"{OUT_JSON}.shard{_si}"
    print(f"Requested {len(tokens)} tokens (shard {_si}/{_ns})")

    traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    config = TransfuserConfig(latent=False)
    agent = TransfuserAgent(
        config=config,
        lr=1e-4,
        checkpoint_path=CKPT,
    )
    agent.initialize()
    agent.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = agent.to(device)
    print(f"Agent loaded on {device}")

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=None,
        tokens=list(tokens),
    )

    loader = SceneLoader(
        data_path=LOGS_PATH,
        sensor_blobs_path=SENSOR_PATH,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    available = set(loader.tokens)
    print(f"Loader matched {len(available)} of {len(tokens)} tokens")
    missing = [t for t in tokens if t not in available]
    if missing:
        print(f"MISSING tokens: {missing}")

    # patch compute_trajectory device handling: move features to device
    results = {}
    n_pts = None
    horizon = traj_sampling.time_horizon
    for tok in tokens:
        if tok not in available:
            continue
        agent_input = loader.get_agent_input_from_token(tok)
        features = {}
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
        with torch.no_grad():
            predictions = agent.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()
        xy = poses[:, :2].tolist()
        n_pts = len(xy)
        results[tok] = xy
        print(f"  {tok}: {len(xy)} pts, first={xy[0]}, last={xy[-1]}")

    out = {
        "_meta": {
            "model": "TransFuser",
            "horizon_s": float(horizon),
            "n_pts": int(n_pts) if n_pts is not None else 0,
            "frame": "ego BEV, x=forward(m), y=left(m), cumulative positions",
            "checkpoint": CKPT,
        }
    }
    for tok in tokens:
        if tok in results:
            out[tok] = results[tok]

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} with {len(results)} tokens (horizon={horizon}s, n_pts={n_pts})")


if __name__ == "__main__":
    main()
