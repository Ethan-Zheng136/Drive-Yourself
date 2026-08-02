#!/usr/bin/env bash
# run_viz_compare.sh -- dump FMHead trajectories for the 0.91 (GOLD) and 0.94 (style-adapter)
# configs on the SAME navtest tokens, then render per-scene side-by-side comparison images.
# Single-GPU (this node exposes one MIG slice). Style axis: LEFT off, RIGHT on (alpha=1).
set -uo pipefail

AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
OUT=/mnt/pfs/zhengguantian/autovla/persona/viz_091_vs_094
TOKENS=$FMHEAD/viz_tokens_240.json

# ---- checkpoints -----------------------------------------------------------
# LEFT = fm2 0.91 (GOLD full-FT v2 head + its paired trained-VLM base; style OFF)
LEFT_HEAD=/mnt/pfs/zhengguantian/autovla/persona/fmhead_GOLD/fullft_fm2_head_pdm0.9088.pt
LEFT_BASE=/mnt/pfs/zhengguantian/autovla/persona/fmhead_GOLD/fullft_fm2_trainedVLM_base.ckpt
# RIGHT = fm2 0.94 (style-adapter head + same full-FT base + persona LoRA, alpha=1, adapter ON)
RIGHT_HEAD=/mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts_style_ddv2/2026-07-16_10-56-07_x4/fmhead_final.pt
RIGHT_BASE=/mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/2026-07-14_15-49-24/full_ft_base.ckpt
PERSONA=/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/ddv2_big_final/lora_final

# ---- shared harness env ----------------------------------------------------
cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export SELECT=pdm FMHEAD_N=16 FMHEAD_STEPS=30

mkdir -p "$OUT/left" "$OUT/right" "$OUT/images"
echo "==== [viz] START $(date '+%F %T') ===="

# ---- 1) LEFT dump (0.91, style OFF, alpha=0) -------------------------------
echo "==== [viz] LEFT dump (fm2 0.91) $(date '+%T') ===="
env -u PERSONA_ADAPTERS \
  BASE_CKPT="$LEFT_BASE" FMHEAD_CKPT="$LEFT_HEAD" \
  ALPHAS=0 FMHEAD_USE_STYLE_ADAPTER=false \
  TOKENS="$TOKENS" OUTDIR="$OUT/left" \
  "$PY" "$FMHEAD/dump_fmhead_style.py" || { echo "[viz][FATAL] LEFT dump failed"; exit 2; }

# ---- 2) RIGHT dump (0.94, style ON, alpha=1, style adapter) ----------------
echo "==== [viz] RIGHT dump (fm2 0.94) $(date '+%T') ===="
PERSONA_ADAPTERS="$PERSONA" \
  BASE_CKPT="$RIGHT_BASE" FMHEAD_CKPT="$RIGHT_HEAD" \
  ALPHAS=1 FMHEAD_USE_STYLE_ADAPTER=true \
  TOKENS="$TOKENS" OUTDIR="$OUT/right" \
  "$PY" "$FMHEAD/dump_fmhead_style.py" || { echo "[viz][FATAL] RIGHT dump failed"; exit 2; }

# ---- 3) render side-by-side images -----------------------------------------
echo "==== [viz] PLOT $(date '+%T') ===="
LEFT_JSON="$OUT/left/persona_a0.json" RIGHT_JSON="$OUT/right/persona_a1.json" \
  OUTDIR="$OUT" IMG_DIR="$OUT/images" \
  "$PY" "$FMHEAD/plot_compare_091_094.py" || { echo "[viz][FATAL] plot failed"; exit 2; }

echo "==== [viz] ALL_DONE $(date '+%F %T') ===="
echo "images: $OUT/images   ($(ls "$OUT/images"/*.png 2>/dev/null | wc -l) png)"
