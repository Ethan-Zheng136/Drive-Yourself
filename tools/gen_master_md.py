#!/usr/bin/env python3
"""Regenerate the human-readable master results table (.md) from the canonical CSV.

The CSV (fm3aligned_master_table.csv) is the single source of truth for all
per-alpha PDMS / YDMS numbers. This script only re-renders it as grouped
markdown; it never edits the CSV. Run it whenever the .md rendering is missing.

Usage:
    python tools/gen_master_md.py \
        --csv /mnt/pfs/zhengguantian/autovla/persona/fm3aligned_master_table.csv \
        --out docs/RESULTS_master_table.md
"""
import argparse
import csv
import datetime as dt
import sys


COLS = [
    ("alpha", "alpha", ""),
    ("PDMS", "PDMS", "{:.4f}"),
    ("DAI", "DAI", "{:.1f}"),
    ("PAI", "PAI", "{:.1f}"),
    ("SAI", "SAI", "{:.1f}"),
    ("v_avg", "v_avg", "{:.2f}"),
    ("peak_acc", "peak_acc", "{:.3f}"),
    ("peak_dec", "peak_dec", "{:.3f}"),
    ("long_jerk", "long_jerk", "{:.3f}"),
    ("lat_jerk", "lat_jerk", "{:.3f}"),
    ("L2->teacher", "L2", "{:.2f}"),
]


def fmt(val, spec):
    if val is None or val == "":
        return "--"
    if not spec:
        return str(val)
    try:
        return spec.format(float(val))
    except (TypeError, ValueError):
        return str(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("empty CSV")

    groups = {}
    order = []
    for r in rows:
        key = (r["scope"], r["teacher"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    header = "| teacher | " + " | ".join(h for _, h, _ in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)

    out = []
    out.append("# fm3aligned master results table")
    out.append("")
    out.append(f"Regenerated from `{args.csv}` on "
               f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    out.append("Canonical source of truth is the CSV; this .md is a rendering only.")
    out.append("")

    last_scope = None
    for (scope, teacher) in order:
        if scope != last_scope:
            out.append("")
            out.append(f"## scope: {scope}")
            out.append("")
            out.append(header)
            out.append(sep)
            last_scope = scope
        for r in groups[(scope, teacher)]:
            cells = [teacher] + [fmt(r.get(src), spec) for src, _, spec in COLS]
            out.append("| " + " | ".join(cells) + " |")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows, {len(order)} groups)")


if __name__ == "__main__":
    main()
