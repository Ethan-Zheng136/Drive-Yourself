#!/usr/bin/env bash
# run_style_diag.sh -- diagnostic style dump (medoid dose-response + pdm@1 washout) then score.
# Read-only w.r.t. checkpoints. All outputs to PFS. Single MIG slice.
set -uo pipefail

AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
OUTDIR_MEDOID=/mnt/pfs/zhengguantian/autovla/persona/fmhead_style_medoid_diag
OUTDIR_PDM=/mnt/pfs/zhengguantian/autovla/persona/fmhead_style_pdm1_diag

cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OUTDIR_MEDOID OUTDIR_PDM
export ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"
export FMHEAD_N="${FMHEAD_N:-16}" FMHEAD_STEPS="${FMHEAD_STEPS:-30}" CFG_WEIGHT="${CFG_WEIGHT:-1.0}"
export LIMIT="${LIMIT:-0}"

echo "==== [diag] DUMP START $(date '+%F %T') ===="
"$PY" "$FMHEAD/diag_style_dump.py" || { echo "[diag][FATAL] dump failed"; exit 2; }

if [ "${LIMIT:-0}" != "0" ]; then
  echo "==== [diag] SMOKE (LIMIT=$LIMIT) done, skipping scoring ===="
  exit 0
fi

echo "==== [diag] SCORE MEDOID $(date '+%T') ===="
"$PY" "$AUTOVLA/youdrive/sweep_metrics.py" --outdir "$OUTDIR_MEDOID" \
  || { echo "[diag][FATAL] medoid scoring failed"; exit 3; }

echo "==== [diag] SCORE PDM@1 $(date '+%T') ===="
"$PY" "$AUTOVLA/youdrive/sweep_metrics.py" --outdir "$OUTDIR_PDM" \
  || { echo "[diag][FATAL] pdm scoring failed"; exit 4; }

echo "==== [diag] ALL_DONE $(date '+%F %T') ===="
