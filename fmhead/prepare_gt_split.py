"""prepare_gt_split.py -- carve a disjoint train/val split from the human-GT navtrain set.

SFTDataset globs *.json in a directory, so we make two symlink dirs (idempotent):
  <out>/train  -> first (N - val_size) scenes
  <out>/val    -> last  val_size scenes
pointing at nocot_navtrain12k. Cheap (symlinks), keeps train/val strictly disjoint.
"""
from __future__ import annotations

import argparse
import glob
import os

SRC = "/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain12k"
OUT = "/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain_gt_split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--val_size", type=int, default=256)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.json")))
    assert files, f"no json in {args.src}"
    val = files[-args.val_size:]
    train = files[:-args.val_size]
    for sub, flist in (("train", train), ("val", val)):
        d = os.path.join(args.out, sub)
        os.makedirs(d, exist_ok=True)
        existing = set(os.listdir(d))
        made = 0
        for f in flist:
            link = os.path.join(d, os.path.basename(f))
            if os.path.basename(f) not in existing:
                os.symlink(f, link)
                made += 1
        print(f"[split] {sub}: {len(flist)} scenes ({made} new symlinks) -> {d}", flush=True)
    print(f"[split] train={len(train)} val={len(val)} (disjoint) under {args.out}", flush=True)


if __name__ == "__main__":
    main()
