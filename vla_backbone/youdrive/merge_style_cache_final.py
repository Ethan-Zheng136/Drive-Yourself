"""Merge the navtrain + navtest DDv2 style caches into one combined cache (CPU).

NON-DESTRUCTIVE: reads two existing *.pkl style caches and writes a NEW combined pkl. Neither
input is modified. The combined cache:
  - 'cache': union of the per-token dicts from both inputs (token-keyed; disjoint splits).
  - 'norm':  RECOMPUTED per-feature (mean, std+1e-6) over the UNION of profile values, so the
             style-reward normalization reflects the combined train+test distribution.
  - 'feats': kept as-is (must be identical across both inputs).

Env:
  IN_TRAIN  navtrain style cache  (default style_cache_navtrain12k.pkl)
  IN_TEST   navtest  style cache  (default style_cache_ddv2_navtest.pkl)
  OUT       combined output       (default style_cache_ddv2_final.pkl)
"""
import os, pickle
import numpy as np

IN_TRAIN = os.environ.get("IN_TRAIN", "/mnt/pfs/zhengguantian/autovla/persona/style_cache_navtrain12k.pkl")
IN_TEST = os.environ.get("IN_TEST", "/mnt/pfs/zhengguantian/autovla/persona/style_cache_ddv2_navtest.pkl")
OUT = os.environ.get("OUT", "/mnt/pfs/zhengguantian/autovla/persona/style_cache_ddv2_final.pkl")
assert OUT.startswith("/mnt/pfs/"), f"OUT must be on PFS, got {OUT}"


def main():
    with open(IN_TRAIN, "rb") as f:
        a = pickle.load(f)
    with open(IN_TEST, "rb") as f:
        b = pickle.load(f)

    feats_a = list(a.get("feats", []))
    feats_b = list(b.get("feats", []))
    if feats_a != feats_b:
        raise ValueError(f"feats mismatch between inputs: {feats_a} vs {feats_b}")
    feats = feats_a

    cache_a = a["cache"]; cache_b = b["cache"]
    overlap = set(cache_a) & set(cache_b)
    if overlap:
        # train/test splits are disjoint; any overlap is unexpected -> fail loud (no silent clobber).
        raise ValueError(f"unexpected token overlap ({len(overlap)}) between train and test caches, e.g. {list(overlap)[:5]}")

    merged = {}
    merged.update(cache_a)
    merged.update(cache_b)

    # recompute per-feature normalization over the UNION (social Nones skipped, matching builders)
    vals = {ftr: [] for ftr in feats}
    for d in merged.values():
        prof = d["profile"]
        for ftr in feats:
            x = prof.get(ftr)
            if x is not None:
                vals[ftr].append(x)
    norm = {ftr: (float(np.mean(vals[ftr])), float(np.std(vals[ftr]) + 1e-6)) for ftr in feats if vals[ftr]}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"cache": merged, "norm": norm, "feats": feats}, f)
    print(f"[merge_style_cache_final] train={len(cache_a)} + test={len(cache_b)} = {len(merged)} tokens -> {OUT}")
    print("norm stats:", {k: (round(m, 2), round(s, 2)) for k, (m, s) in norm.items()})


if __name__ == "__main__":
    main()
