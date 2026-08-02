#!/usr/bin/env bash
# eval_fm3aligned_batch.sh -- run eval_fm3aligned_teacher.sh for gtrs / hydra_mdp / wote (or ALL six).
#
# Usage:
#   bash /root/workspace/fmhead/eval_fm3aligned_batch.sh new3
#   bash /root/workspace/fmhead/eval_fm3aligned_batch.sh all
#   GPUS="0 1 2 3" LIMIT=400 ALPHAS="0,1" bash eval_fm3aligned_batch.sh new3
set -euo pipefail

FMHEAD=/root/workspace/fmhead
MODE="${1:-new3}"

run_one() {
  local teacher="$1" lora="$2" out="$3"
  echo "======== batch: $teacher ========"
  TEACHER="$teacher" LORA="$lora" OUTDIR="$out" bash "$FMHEAD/eval_fm3aligned_teacher.sh"
}

PFS=/mnt/pfs/zhengguantian/autovla/persona

case "$MODE" in
  new3)
    run_one gtrs \
      "$PFS/lora_fm3aligned_gtrs/2026-07-23_18-13-26_x8/lora_final" \
      "$PFS/sweep_fm3aligned_gtrs"
    run_one hydra_mdp \
      "$PFS/lora_fm3aligned_hydra_mdp/2026-07-23_18-13-31_x8/lora_final" \
      "$PFS/sweep_fm3aligned_hydra_mdp"
    run_one wote \
      "$PFS/lora_fm3aligned_wote/2026-07-23_19-25-16_x8/lora_final" \
      "$PFS/sweep_fm3aligned_wote"
    ;;
  all)
    run_one ddv2 \
      "$PFS/lora_fm3aligned_ddv2/2026-07-20_02-50-59_x8/lora_final" \
      "$PFS/sweep_fm3aligned_ddv2"
    run_one goalflow \
      "$PFS/lora_fm3aligned_goalflow/2026-07-21_02-13-42_x8/lora_final" \
      "$PFS/sweep_fm3aligned_goalflow"
    run_one transfuser \
      "$PFS/lora_fm3aligned_transfuser/2026-07-20_02-51-39_x8/lora_final" \
      "$PFS/sweep_fm3aligned_transfuser"
    run_one gtrs \
      "$PFS/lora_fm3aligned_gtrs/2026-07-23_18-13-26_x8/lora_final" \
      "$PFS/sweep_fm3aligned_gtrs"
    run_one hydra_mdp \
      "$PFS/lora_fm3aligned_hydra_mdp/2026-07-23_18-13-31_x8/lora_final" \
      "$PFS/sweep_fm3aligned_hydra_mdp"
    run_one wote \
      "$PFS/lora_fm3aligned_wote/2026-07-23_19-25-16_x8/lora_final" \
      "$PFS/sweep_fm3aligned_wote"
    ;;
  *)
    echo "Usage: bash $0 {new3|all}"; exit 2 ;;
esac

echo "[batch] ALL_DONE mode=$MODE"
