"""Quantization round-trip diagnostic.

Feeds the (continuous) DDv2 teacher trajectories through AutoVLA's codebook
tokenizer and measures how much the discrete codebook distorts them (ADE/FDE).

Tiny error  => codebook is near-lossless => imitation ceiling ~= teacher PDMS
              => the alpha=1 gap is caused by training (forgetting/BC), not representation.
Large error => codebook cannot express the teacher's trajectories
              => representation is a real ceiling; SFT alone cannot recover it.

Usage:
  python youdrive/quant_ceiling.py [N_SAMPLE]
"""
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from omegaconf import DictConfig

from navsim.agents.autovla_agent import TokenProcessor

CODEBOOK = "codebook_cache/agent_vocab.pkl"
TEACHER_DIR = "/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k"


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

    tp = TokenProcessor(
        agent_token_file=CODEBOOK,
        agent_token_sampling=DictConfig({"num_k": 1, "temp": 1.0}),
    )
    tp.eval()  # training=False => num_k=1 => deterministic nearest-token snap

    files = sorted(glob.glob(f"{TEACHER_DIR}/*.json"))
    if not files:
        print(f"no teacher json under {TEACHER_DIR}")
        return
    random.seed(0)
    if n_sample and n_sample < len(files):
        files = random.sample(files, n_sample)

    ades, fdes, maxes = [], [], []
    n_ok = 0
    for fp in files:
        try:
            d = json.load(open(fp))
            traj = d.get("gt_trajectory")
            if not traj:
                continue
            arr = np.asarray(traj, dtype=np.float32)  # [T, 3] = x,y,heading
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
                continue
            data = torch.from_numpy(arr)  # [T, 3]
            with torch.no_grad():
                out = tp(data)
            q = out["gt_pos"][0].cpu().numpy()  # [T, 2] quantized positions
            gt = arr[: q.shape[0], :2]
            err = np.linalg.norm(q - gt, axis=-1)  # [T]
            ades.append(float(err.mean()))
            fdes.append(float(err[-1]))
            maxes.append(float(err.max()))
            n_ok += 1
        except Exception as e:
            print(f"  skip {os.path.basename(fp)}: {e}")
            continue

    if not ades:
        print("no valid trajectories scored")
        return

    ades = np.asarray(ades)
    fdes = np.asarray(fdes)
    maxes = np.asarray(maxes)

    def pct(a, p):
        return float(np.percentile(a, p))

    print(f"\n===== codebook quantization round-trip on DDv2 teacher (n={n_ok}) =====")
    print(f"  ADE (mean per-step displacement error, meters)")
    print(f"     mean={ades.mean():.3f}  median={np.median(ades):.3f}  p90={pct(ades,90):.3f}  p99={pct(ades,99):.3f}  max={ades.max():.3f}")
    print(f"  FDE (final point error, meters)")
    print(f"     mean={fdes.mean():.3f}  median={np.median(fdes):.3f}  p90={pct(fdes,90):.3f}  max={fdes.max():.3f}")
    print(f"  per-traj worst-step error")
    print(f"     mean={maxes.mean():.3f}  p90={pct(maxes,90):.3f}  max={maxes.max():.3f}")


if __name__ == "__main__":
    main()
