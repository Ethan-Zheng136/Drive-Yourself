#!/usr/bin/env bash
# run_viz_4panel_fm3kin.sh -- render the 2x2 four-panel per-scene comparison
# fm2 0.91 (GOLD, LEFT) vs fm3-kin ep6 (PDMS 0.915, RIGHT). Top = my-style ego-BEV traj,
# bottom = navsim classic BEV. LEFT reuses the existing 0.91 dump; RIGHT reads the fm3-kin
# ep6 kinematic dump (dump_fm3kin_viz.py). Uses plot_4panel_091_vs_fm3kin.py (kinematic
# captions + per-panel jerk). CPU-only (matplotlib + navsim map render).
set -uo pipefail
AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
OUT=/mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_vs_fm3kin_ep6

cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

export LEFT_JSON="${LEFT_JSON:-/mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094/left/persona_a0.json}"
export RIGHT_JSON="${RIGHT_JSON:-$OUT/right/fm3kin_ep6.json}"
export TOKENS="$FMHEAD/viz_tokens_240.json"
export N_TOKENS="${N_TOKENS:-200}"
export OUTDIR="${OUTDIR:-$OUT}"
export IMG_DIR="${IMG_DIR:-$OUTDIR/images}"
export JSON_DIR="$AUTOVLA/dataset/nuplan/navtest_nocot"
export DPI="${DPI:-140}"
export MONTAGE_N="${MONTAGE_N:-12}"
export LEFT_LABEL="${LEFT_LABEL:-fm2 0.91}"
export RIGHT_LABEL="${RIGHT_LABEL:-fm3-kin ep6 (0.915)}"

mkdir -p "$IMG_DIR"
echo "==== [4panel fm3-kin] START $(date '+%F %T') | N_TOKENS=$N_TOKENS DPI=$DPI ===="
echo "     LEFT=$LEFT_JSON"
echo "     RIGHT=$RIGHT_JSON"
echo "     out=$OUTDIR"
"$PY" "$FMHEAD/plot_4panel_091_vs_fm3kin.py"
rc=$?
echo "==== [4panel fm3-kin] EXIT rc=$rc $(date '+%F %T') ===="
echo "images: $IMG_DIR   ($(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l) png)"
exit $rc
