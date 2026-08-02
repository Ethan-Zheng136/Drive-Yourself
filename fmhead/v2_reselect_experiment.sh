#!/usr/bin/env bash
# v2_reselect_experiment.sh -- NEW, additive. Runs OFFLINE (CPU) on a box with the v2 navsim
# env + the navtest v2 metric cache. Given a directory of dumped candidates (from the agent's
# dump_all mode), it scores EACH of the N candidate slots through the v2 EPDMS pipeline and
# reports the selection headroom (per-candidate means + oracle ceiling). CPU-only, no GPU.
#
# The scoring reuses score_existing_submission.sh; the aggregation (agg_reselect.py) recomputes
# the CORRECT single-stage EPDMS excluding two_frame -> the 0.7 bug can NOT appear here.
#
# Usage:
#   BASELINE=0.837 bash v2_reselect_experiment.sh <candidates_dir> <out_dir> [N]
set -uo pipefail
CAND_DIR="${1:?usage: $0 <candidates_dir> <out_dir> [N]}"
OUT="${2:?usage: $0 <candidates_dir> <out_dir> [N]}"
N="${3:-16}"

V2CACHE=/root/workspace/closed_loop/data/navsim/exp/metric_cache_navtest
PY=/root/workspace/miniconda3/envs/navsim/bin/python
SCORE=/root/workspace/closed_loop/navsim_integration/score_existing_submission.sh
FMHEAD=/root/workspace/fmhead
mkdir -p "$OUT"

for k in $(seq 0 $((N - 1))); do
  d="$OUT/sub$k"; mkdir -p "$d"
  echo "==================== candidate $k / $((N-1)): build + score ===================="
  if ! "$PY" "$FMHEAD/build_kth_submission.py" "$CAND_DIR" "$k" "$d/submission.pkl"; then
    echo "  [warn] build failed for k=$k, skipping"; continue
  fi
  NAVSIM_DATA=/root/workspace/closed_loop/data/navsim bash "$SCORE" "$d/submission.pkl" navtest "$V2CACHE"
done

echo "==================== aggregate ===================="
"$PY" "$FMHEAD/agg_reselect.py" "$OUT" "$N" "${BASELINE:-}"
