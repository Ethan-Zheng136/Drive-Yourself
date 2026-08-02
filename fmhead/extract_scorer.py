"""extract_scorer.py -- pull the deployable FMHeadScorer out of a JOINT Lightning ckpt.

fmhead_sft_joint.py saves the scorer INSIDE the Lightning checkpoint (ModelCheckpoint,
state_dict_type='full') under keys `scorer.*`. This extracts those into a standalone
scorer_final.pt with the EXACT {'scorer': state_dict, 'config': {...}} format that
fmhead_navsim_agent.py (mode=learned) expects, reconstructing the scorer config from the
same joint yaml used for training.

Usage:
  python extract_scorer.py --ckpt <epoch=*.ckpt> --config config/fmhead_fm3_kin_joint.yaml \
      --out /mnt/pfs/.../fmhead_fm3_kin_joint/<ts>/scorer_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)

from fmhead_scorer import V1_METRICS, V2_EXTRA_METRICS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Lightning epoch=*.ckpt from the joint run")
    ap.add_argument("--config", required=True, help="the joint yaml used for training")
    ap.add_argument("--out", required=True, help="output scorer_final.pt (agent mode=learned format)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    j = cfg.get("joint_scorer") or {}
    metrics = list(V1_METRICS)
    if str(j.get("metrics", "v1")).lower() == "v2":
        metrics = list(V1_METRICS) + list(V2_EXTRA_METRICS)
    sc_config = {
        "env_in_dim": int(j.get("env_in_dim", 2048)),
        "d_model": int(j.get("d_model", 256)),
        "n_decoder_layers": int(j.get("n_decoder_layers", 3)),
        "num_poses": int(cfg["model"]["trajectory"]["num_poses"]),
        "metrics": metrics,
    }

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    scorer_sd = {k[len("scorer."):]: v for k, v in sd.items() if k.startswith("scorer.")}
    if not scorer_sd:
        raise SystemExit(f"[extract_scorer] no `scorer.*` keys in {args.ckpt} -- was joint_scorer "
                         f"enabled during training?")
    # cast bf16 (training dtype) -> fp32 for a portable deployable scorer
    scorer_sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in scorer_sd.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"scorer": scorer_sd, "config": sc_config}, args.out)
    print(f"[extract_scorer] wrote {args.out}  ({len(scorer_sd)} tensors, config={sc_config})",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
