#!/usr/bin/env bash
# eval_fm3aligned_teacher.sh -- YDMS (styletest) + PDMS (navtest) for one fm3-aligned persona LoRA.
#
# Uses latent content interpolation (FMHEAD_CONTENT_SOURCE=interp): c = h_base + alpha*(h_styled-h_base).
#
# Usage:
#   TEACHER=gtrs \
#   LORA=/mnt/pfs/.../lora_fm3aligned_gtrs/.../lora_final \
#   OUTDIR=/mnt/pfs/.../sweep_fm3aligned_gtrs \
#   bash /root/workspace/fmhead/eval_fm3aligned_teacher.sh
#
#   # positional (same order):
#   bash eval_fm3aligned_teacher.sh gtrs /path/lora_final /path/outdir
#
# Smoke (1 GPU, 400 styletest scenes, alphas 0+1 only):
#   GPUS="0" LIMIT=400 ALPHAS="0,1" PDMS_ALPHAS="0,1" SUBSET_N=400 \
#     TEACHER=gtrs LORA=... OUTDIR=... bash eval_fm3aligned_teacher.sh
#
# Env:
#   TEACHER     sweep_metrics --teacher (ddv2|transfuser|goalflow|gtrs|hydra_mdp|wote)
#   LORA        persona LoRA dir (lora_final)
#   OUTDIR      PFS output root
#   GPUS        default "0 1 2 3 4 5 6 7"
#   LIMIT       styletest token cap; 0 = full styletest (default 0)
#   ALPHAS      style dump alphas (default 0,0.25,0.5,0.75,1.0)
#   RUN_PDMS    1 (default) | 0 skip navtest PDMS
#   PDMS_ALPHAS subset for PDMS only (default = ALPHAS)
#   SKIP_DUMP   1 = only sweep+pdms (dumps already in OUTDIR)
#   SKIP_SWEEP  1 = skip YDMS sweep
set -euo pipefail

FMHEAD=/root/workspace/fmhead
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
PY=/root/workspace/miniconda3/envs/autovla/bin/python

TEACHER="${TEACHER:-${1:-}}"
LORA="${LORA:-${2:-}}"
OUTDIR="${OUTDIR:-${3:-}}"

if [ -z "$TEACHER" ] || [ -z "$LORA" ] || [ -z "$OUTDIR" ]; then
  echo "Usage: TEACHER=<name> LORA=<lora_final_dir> OUTDIR=<pfs_dir> bash $0"
  echo "   or: bash $0 <teacher> <lora_final_dir> <outdir>"
  echo ""
  echo "Known LoRA paths (lora_final):"
  echo "  gtrs      /mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_gtrs/2026-07-23_18-13-26_x8/lora_final"
  echo "  hydra_mdp /mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_hydra_mdp/2026-07-23_18-13-31_x8/lora_final"
  echo "  wote      /mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_wote/2026-07-23_19-25-16_x8/lora_final"
  echo "  ddv2      /mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_ddv2/2026-07-20_02-50-59_x8/lora_final"
  echo "  goalflow  /mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_goalflow/2026-07-21_02-13-42_x8/lora_final"
  echo "  transfuser /mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_transfuser/2026-07-20_02-51-39_x8/lora_final"
  exit 2
fi

[ -d "$LORA" ] || { echo "[eval] missing LORA dir: $LORA"; exit 2; }

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LIMIT="${LIMIT:-0}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"
PDMS_ALPHAS="${PDMS_ALPHAS:-$ALPHAS}"
RUN_PDMS="${RUN_PDMS:-1}"
SELECT="${SELECT:-pdm}"
SUBSET_N="${SUBSET_N:-0}"

KIN=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42
HEAD="${FMHEAD_CKPT:-$KIN/fullft_fm3_kin_ep6_head.pt}"
BASE="${BASE_CKPT:-$KIN/fullft_fm3_kin_ep6_base.ckpt}"
CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs
mkdir -p "$OUTDIR" "$LOG"

echo "[eval] teacher=$TEACHER lora=$LORA outdir=$OUTDIR"
echo "[eval] gpus=$GPUS limit=$LIMIT alphas=$ALPHAS run_pdms=$RUN_PDMS"

# ---- (1) styletest dump (interp alpha sweep) ----
if [ "${SKIP_DUMP:-0}" != "1" ]; then
  GPUS="$GPUS" LIMIT="$LIMIT" ALPHAS="$ALPHAS" OUTDIR="$OUTDIR" \
    SELECT="$SELECT" SKIP_SWEEP=1 \
    PERSONA_ADAPTERS="$LORA" \
    FMHEAD_CKPT="$HEAD" BASE_CKPT="$BASE" \
    bash "$FMHEAD/dump_latent_interp.sh"
else
  echo "[eval] SKIP_DUMP=1 -> reusing dumps in $OUTDIR"
fi

# ---- (2) YDMS / style_metrics (correct teacher anchor for L2) ----
if [ "${SKIP_SWEEP:-0}" != "1" ]; then
  "$PY" "$AV/youdrive/sweep_metrics.py" --outdir "$OUTDIR" --teacher "$TEACHER"
  echo "[eval] YDMS -> $OUTDIR/style_metrics.txt"
else
  echo "[eval] SKIP_SWEEP=1"
fi

# ---- (3) navtest PDMS per alpha ----
if [ "$RUN_PDMS" = "1" ]; then
  : > "$OUTDIR/pdms_summary.txt"
  echo "# teacher=$TEACHER lora=$LORA" >> "$OUTDIR/pdms_summary.txt"
  cd "$FMHEAD"
  IFS=',' read -r -a _ALPHAS <<< "$PDMS_ALPHAS"
  for a in "${_ALPHAS[@]}"; do
  a="$(echo "$a" | tr -d ' ')"
  TAG="fm3align_${TEACHER}_a${a}"
  echo "[eval] PDMS alpha=$a tag=$TAG"
  export PERSONA_ADAPTERS="$LORA"
  SUBSET_N="$SUBSET_N" \
    DECODER=fmhead TAG="$TAG" \
    FMHEAD_CKPT="$HEAD" \
    BASE_CKPT="$BASE" \
    FMHEAD_NORM="$FMHEAD/traj_norm_stats_ctrl.json" \
    FMHEAD_CONTENT_SOURCE=interp \
    FMHEAD_ALPHA="$a" \
    FMHEAD_CONSTRAINT="$CONSTRAINT" \
    SELECT="$SELECT" \
    GPUS="$GPUS" \
    bash run_pdms_fmhead.sh \
    | tee "$LOG/pdms_${TAG}.log" \
    | tee -a "$OUTDIR/pdms_summary.txt"
  done
  echo "[eval] PDMS summary -> $OUTDIR/pdms_summary.txt"
fi

echo "[eval] ALL_DONE teacher=$TEACHER -> $OUTDIR"
