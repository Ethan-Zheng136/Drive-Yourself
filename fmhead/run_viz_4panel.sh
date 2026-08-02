#!/usr/bin/env bash
# run_viz_4panel.sh -- render the 2x2 four-panel per-scene comparison (fm2 0.91 vs 0.94).
# Top row = my-style ego-BEV traj; bottom row = navsim classic BEV. REUSES the existing
# trajectory dumps from viz_091_vs_094 (NO inference). CPU-only (matplotlib + navsim map render).
set -uo pipefail

AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
PREV=/mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094
OUT=/mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_094

cd "$AUTOVLA"
# ---- navsim / nuplan data roots (from youdrive/eval_persona.sh) ----
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# ---- viz config ----
export LEFT_JSON="$PREV/left/persona_a0.json"
export RIGHT_JSON="$PREV/right/persona_a1.json"
export TOKENS="$FMHEAD/viz_tokens_240.json"
export N_TOKENS="${N_TOKENS:-200}"
export OUTDIR="${OUTDIR:-$OUT}"
export IMG_DIR="${IMG_DIR:-$OUTDIR/images}"
export JSON_DIR="$AUTOVLA/dataset/nuplan/navtest_nocot"
export DPI="${DPI:-140}"
export MONTAGE_N="${MONTAGE_N:-12}"

mkdir -p "$IMG_DIR"
echo "==== [4panel] START $(date '+%F %T') | N_TOKENS=$N_TOKENS ===="
echo "     out=$OUT"
"$PY" "$FMHEAD/plot_4panel_091_094.py"
rc=$?
echo "==== [4panel] EXIT rc=$rc $(date '+%F %T') ===="
echo "images: $IMG_DIR   ($(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l) png)"
exit $rc
