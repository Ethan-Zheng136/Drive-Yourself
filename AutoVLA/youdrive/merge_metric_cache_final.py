"""Merge the navtrain12k + navtest_v1 metric caches into one loadable dir (CPU, non-destructive).

navsim's MetricCacheLoader is driven ENTIRELY by a single metadata CSV: it reads
  <cache_path>/metadata/<first *.csv>
whose rows are ABSOLUTE paths to each token's metric_cache.pkl, and keys them by token
(= path.split('/')[-2]). It never walks the cache tree. So a clean, safe merge is simply a NEW
dir whose metadata/ holds ONE CSV = header + all rows of BOTH source CSVs. The rows still point at
the ORIGINAL pkls (absolute paths), so nothing is copied and neither source cache is touched.

Env:
  IN_TRAIN  navtrain metric cache dir (default exp/metric_cache_navtrain12k)
  IN_TEST   navtest  metric cache dir (default exp/metric_cache_navtest_v1)
  OUT       combined metric cache dir (default exp/metric_cache_final)
"""
import os, sys

IN_TRAIN = os.environ.get("IN_TRAIN", "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtrain12k")
IN_TEST = os.environ.get("IN_TEST", "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
OUT = os.environ.get("OUT", "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_final")
assert OUT.startswith("/mnt/pfs/"), f"OUT must be on PFS, got {OUT}"


def read_csv_rows(cache_dir):
    meta_dir = os.path.join(cache_dir, "metadata")
    csvs = [f for f in os.listdir(meta_dir) if f.endswith(".csv")]
    if not csvs:
        sys.exit(f"ERROR: no metadata csv in {meta_dir}")
    path = os.path.join(meta_dir, sorted(csvs)[0])
    with open(path) as f:
        lines = f.read().splitlines()
    header, rows = lines[0], [r for r in lines[1:] if r.strip()]
    return header, rows, path


def token_of(row):
    return row.split("/")[-2]


def main():
    h1, r1, p1 = read_csv_rows(IN_TRAIN)
    h2, r2, p2 = read_csv_rows(IN_TEST)
    if h1 != h2:
        sys.exit(f"ERROR: metadata csv header mismatch: {h1!r} vs {h2!r}")

    t1 = {token_of(r): r for r in r1}
    t2 = {token_of(r): r for r in r2}
    overlap = set(t1) & set(t2)
    if overlap:
        # disjoint splits expected; fail loud rather than silently dropping duplicates.
        sys.exit(f"ERROR: unexpected token overlap ({len(overlap)}) between caches, e.g. {list(overlap)[:5]}")

    merged = {**t1, **t2}
    os.makedirs(os.path.join(OUT, "metadata"), exist_ok=True)
    out_csv = os.path.join(OUT, "metadata", "metric_cache_final_metadata_node_0.csv")
    with open(out_csv, "w") as f:
        f.write(h1 + "\n")
        for tok in sorted(merged):
            f.write(merged[tok] + "\n")
    print(f"[merge_metric_cache_final] train={len(t1)} + test={len(t2)} = {len(merged)} tokens")
    print(f"  from:\n    {p1}\n    {p2}")
    print(f"  wrote -> {out_csv}")


if __name__ == "__main__":
    main()
