#!/usr/bin/env bash
# One-shot GRPO verification for a driver: INFER (navtest PDMS) + SWEEP (styletest jerk style).
#
# Evaluates TWO checkpoints (both DE-MERGE on the raw AutoVLA_PDMS_89 base):
#   * GRPO    = base + Stage-1 LoRA (frozen) + Stage-2 LoRA (the GRPO output)
#   * persona = base + Stage-1 LoRA only  (the pre-GRPO SFT baseline)
# sweep_metrics.py additionally anchors base(AutoVLA) / DDv2 teacher / human automatically,
# so the style table is a full side-by-side.
#
# Usage:
#   bash youdrive/verify_grpo.sh ddv2
#   bash youdrive/verify_grpo.sh goalflow
#   bash youdrive/verify_grpo.sh transfuser
# Options (env):
#   PERSONA=0   skip the SFT-persona baseline (only verify the GRPO checkpoint)
#   GPUS="0 1 2 3 4 5 6 7"   GPUs to shard across (passed through to the inner scripts)
set -uo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AV"
PY=/root/workspace/miniconda3/envs/autovla/bin/python
LCK=/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs
CMP=/mnt/pfs/zhengguantian/autovla

DRIVER="${1:?usage: [VARIANT=big|huge] verify_grpo.sh <ddv2|goalflow|transfuser>}"
VARIANT="${VARIANT:-big}"          # big = r64 stage1 + *-grpo.log ; huge = r128 stage1 + *-grpo-huge.log
DO_PERSONA="${PERSONA:-1}"

# --- per-(driver,variant) Stage-1 (SFT persona) adapter + log suffix + output tag ---
case "$VARIANT" in
  big)
    case "$DRIVER" in
      ddv2)                 STAGE1=$LCK/big_l2_r64_20260623/lora_final ;;
      goalflow)             STAGE1=$LCK/goalflow_fixed_big_l2_r64/lora_final ;;
      transfuser)           STAGE1=$LCK/transfuser_big_l2_r64/lora_final ;;
      gtrs|hydra_mdp|wote)  STAGE1=$LCK/${DRIVER}_big_l2_r64/lora_final ;;
      *) echo "[verify] unknown driver: $DRIVER"; exit 1 ;;
    esac
    GSUF=""; VT="" ;;
  huge)
    case "$DRIVER" in
      ddv2|goalflow|transfuser) STAGE1=$LCK/${DRIVER}_huge_l2_r128/lora_final ;;
      *) echo "[verify] unknown driver: $DRIVER (expected ddv2|goalflow|transfuser)"; exit 1 ;;
    esac
    GSUF="-huge"; VT="_huge" ;;
  *) echo "[verify] unknown VARIANT: $VARIANT (expected big|huge)"; exit 1 ;;
esac
if [ ! -d "$STAGE1" ]; then echo "[verify] FATAL stage1 not found: $STAGE1"; exit 1; fi

# --- locate the GRPO Stage-2 adapter from the driver's training log (prefer lora_final) ---
GLOG=$LOG/train_grpo_youdrive-$DRIVER-grpo$GSUF.log
if [ ! -f "$GLOG" ]; then echo "[verify] FATAL grpo log not found: $GLOG"; exit 1; fi
STAGE2=$(grep -a "saved STAGE2 LoRA adapter" "$GLOG" | grep -a "lora_final/stage2" | tail -1 | sed 's/.*-> //')
if [ -z "$STAGE2" ]; then
  STAGE2=$(grep -a "saved STAGE2 LoRA adapter" "$GLOG" | tail -1 | sed 's/.*-> //')
  echo "[verify] WARNING lora_final not found yet -> using latest checkpoint: $STAGE2"
fi
if [ -z "$STAGE2" ] || [ ! -d "$STAGE2" ]; then
  echo "[verify] FATAL Stage-2 adapter not found (has the run finished?): '$STAGE2'"; exit 1
fi

echo "[verify] ============================================"
echo "[verify] driver = $DRIVER  variant = $VARIANT"
echo "[verify] stage1 = $STAGE1"
echo "[verify] stage2 = $STAGE2"
echo "[verify] persona baseline = $([ "$DO_PERSONA" = 1 ] && echo yes || echo skip)"
echo "[verify] ============================================"

# ===================== A. INFER: navtest PDMS (safety) =====================
echo "[verify] ===== [A] INFER navtest PDMS ====="
ADAPTERS=$STAGE1,$STAGE2 WEIGHTS=1.0,1.0 TAG=${DRIVER}_grpo$VT bash youdrive/eval_persona.sh
if [ "$DO_PERSONA" = 1 ]; then
  ADAPTERS=$STAGE1 WEIGHTS=1.0 TAG=${DRIVER}_persona$VT bash youdrive/eval_persona.sh
fi

# ===================== B. SWEEP: styletest jerk style =====================
echo "[verify] ===== [B] SWEEP styletest jerk style ====="
OG=$CMP/compare_${DRIVER}_grpo$VT
ADAPTER=$STAGE1,$STAGE2 WEIGHTS=1,1 ALPHAS=1.0 OUTDIR=$OG bash youdrive/dump_persona_sweep_8gpu.sh
$PY youdrive/sweep_metrics.py --outdir "$OG"

if [ "$DO_PERSONA" = 1 ]; then
  OP=$CMP/compare_${DRIVER}_persona$VT
  ADAPTER=$STAGE1 WEIGHTS=1 ALPHAS=1.0 OUTDIR=$OP bash youdrive/dump_persona_sweep_8gpu.sh
  $PY youdrive/sweep_metrics.py --outdir "$OP"
fi

echo "[verify] ============================================"
echo "[verify] DONE $DRIVER"
echo "[verify] navtest PDMS  : $LOG/eval_${DRIVER}_grpo${VT}_shard*.log  (agg printed above)"
[ "$DO_PERSONA" = 1 ] && echo "[verify]               + $LOG/eval_${DRIVER}_persona${VT}_shard*.log"
echo "[verify] style table   : $OG/style_metrics.txt"
[ "$DO_PERSONA" = 1 ] && echo "[verify]               + $CMP/compare_${DRIVER}_persona$VT/style_metrics.txt"
echo "[verify] ============================================"
