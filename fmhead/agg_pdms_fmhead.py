"""agg_pdms_fmhead.py -- aggregate navtest PDMS shard CSVs for the FMHead feasibility run.

Why this exists: `youdrive/_agg_pdms.py` (in the AutoVLA repo, which we must NOT modify)
hardcodes a `persona_{tag}_shard{i}/` experiment-dir glob. `run_pdms_fmhead.sh` writes
`experiment_name=fmfeas_{tag}_shard{i}`, so the stock aggregator finds no CSVs and prints
"no scores" with all-nan metrics -- even though every shard produced valid scores. This
aggregator globs the correct `fmfeas_` prefix and averages the EPDMS score + sub-metrics.

Usage: python agg_pdms_fmhead.py <TAG> <NSHARD> [exp_root] [prefix] [--require-all]

--require-all : REFUSE to print a trusted PDMS unless ALL NSHARD shards produced a CSV.
                If any shard is missing, print a loud 'PARTIAL k/N -- DO NOT TRUST'
                banner and exit non-zero (prevents silent misleading partial numbers,
                e.g. the 0.28-on-6066/12133 half-success we hit when GPUs went missing).
"""
import csv
import glob
import sys

argv = [a for a in sys.argv[1:] if a != "--require-all"]
require_all = "--require-all" in sys.argv
tag = argv[0]
nshard = int(argv[1])
root = argv[2] if len(argv) > 2 else "/mnt/pfs/zhengguantian/autovla/exp"
prefix = argv[3] if len(argv) > 3 else "fmfeas"

rows = []
per_shard = []
n_with_csv = 0
for i in range(nshard):
    cs = sorted(glob.glob(f"{root}/{prefix}_{tag}_shard{i}/*/*.csv"))
    if not cs:
        print(f"[warn] shard{i}: no CSV under {root}/{prefix}_{tag}_shard{i}/*/*.csv")
        per_shard.append((i, 0))
        continue
    r = list(csv.DictReader(open(cs[-1])))
    rows += r
    per_shard.append((i, len(r)))
    n_with_csv += 1

partial = n_with_csv < nshard
if partial and require_all:
    print(f"\n===== PARTIAL {n_with_csv}/{nshard} shards -- DO NOT TRUST =====")
    print("    Missing shards produced no CSV (crash / GPU missing). Refusing to report a")
    print("    PDMS number on an incomplete scene set. Re-run the failed shards.")
    for i, k in per_shard:
        print(f"    shard{i}: {'MISSING' if k == 0 else str(k)+' rows'}")
    sys.exit(2)


def col_mean(col, valid_only=True):
    vals = []
    for r in rows:
        if valid_only and str(r.get("valid", "True")).lower() in ("false", "0"):
            continue
        v = r.get(col)
        if v not in (None, ""):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


score, n = col_mean("score")
banner = f"===== FMHead[{tag}] navtest PDMS = {score:.4f}  (n={n} valid / {len(rows)} rows) ====="
if partial:
    banner = f"===== [PARTIAL {n_with_csv}/{nshard} shards - DO NOT TRUST] " + banner[6:]
print(f"\n{banner}")
for i, k in per_shard:
    print(f"    shard{i}: {k} rows")
print("    --- sub-metrics (mean over valid) ---")
for col in [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
    "driving_direction_compliance",
]:
    m, k = col_mean(col)
    print(f"    {col:34s} {m:.4f}  (n={k})")
