#!/usr/bin/env python
"""agg_reselect.py -- NEW, additive.

Aggregate the per-candidate v2 EPDMS CSVs produced by v2_reselect_experiment.sh into a
token x candidate score matrix, using the CORRECT single-stage EPDMS (multiplicative_prod x
weighted_avg over {EP,TTC,LK,HC}, EXCLUDING two_frame_extended_comfort -- i.e. reproducing
PDMScorer._aggregate_pdm_scores). This is the same correction that fixes the 0.7 reporting
bug, so no number here is contaminated by two_frame=0.

Reports:
  * per-candidate mean EPDMS* (the model's raw sampling quality per slot)
  * mean over candidates (~ picking a random single sample)
  * CEILING = mean_t max_k  (oracle selection = the UPPER BOUND any selector can reach;
    DIAGNOSTIC ONLY, not a reportable result -- it uses the eval metric to pick)
  * WORST   = mean_t min_k

Usage:
  python agg_reselect.py <out_dir> <N> [baseline_v1select]
"""
import sys, glob
import pandas as pd
import numpy as np

W = dict(p=5.0, t=5.0, lk=2.0, hc=2.0)
DEN = sum(W.values())


def corrected_series(csv):
    df = pd.read_csv(csv)
    per = df[~df["token"].astype(str).str.startswith("extended_pdm_score_")].copy()
    v = per[per["valid"].astype(str).str.lower().eq("true")].copy()
    g = lambda c: pd.to_numeric(v[c], errors="coerce")
    mult = g("multiplicative_metrics_prod_stage_one")
    wavg = (W['p'] * g("ego_progress_stage_one")
            + W['t'] * g("time_to_collision_within_bound_stage_one")
            + W['lk'] * g("lane_keeping_stage_one")
            + W['hc'] * g("history_comfort_stage_one")) / DEN
    s = (mult * wavg)
    s.index = v["token"].values
    return s


def main():
    out_dir, N = sys.argv[1], int(sys.argv[2])
    baseline = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("", "None") else None
    cols = {}
    for k in range(N):
        cs = sorted(glob.glob(f"{out_dir}/sub{k}/pdm_score_*/*.csv"))
        if not cs:
            print(f"  [warn] missing CSV for candidate k={k}")
            continue
        cols[k] = corrected_series(cs[-1])
    if not cols:
        raise SystemExit("ERROR: no candidate CSVs found")
    M = pd.DataFrame(cols).dropna(how="any")
    print(f"\n==================== v2 re-selection headroom (navtest one-stage EPDMS*) ====================")
    print(f"  tokens (all-candidate valid): {len(M)}   candidates: {M.shape[1]}")
    per_k = M.mean(axis=0)
    print("\n  per-candidate mean EPDMS*:")
    print("    " + "  ".join(f"k{k}:{per_k[k]:.4f}" for k in per_k.index))
    print(f"\n  mean over candidates (~random single sample): {per_k.mean():.4f}")
    print(f"  WORST   (mean_t min_k):                       {M.min(axis=1).mean():.4f}")
    print(f"  CEILING (oracle mean_t max_k, DIAGNOSTIC):    {M.max(axis=1).mean():.4f}")
    if baseline is not None:
        print(f"  current v1-PDM-select baseline:               {baseline:.4f}")
        print(f"  => selection headroom above current:          {M.max(axis=1).mean() - baseline:+.4f}")
    print("  NOTE: CEILING uses the eval metric to pick -> upper bound only, NOT a reportable")
    print("        number. A reportable v2-select needs a surrogate/deployable scorer.")
    print("=" * 92)


if __name__ == "__main__":
    main()
