#!/usr/bin/env bash
# Train DDv2-persona Style-LoRA on AutoVLA (single-node, all visible GPUs / DDP).
# Usage: bash youdrive/train_ddv2_persona.sh
# Optional env:
#   PERSONA_KL_BETA=20   # KL anchor to frozen base policy (0=pure imitation, original behavior)
#   SAVE_DIR=/path       # where adapters go (default: timestamped under persona/lora_ckpts)
#   SAVE_EVERY=400       # save intermediate LoRA every N optimizer steps
set -uo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AV"
export PYTHONPATH="$AV:$AV/navsim:${PYTHONPATH:-}"
export PERSONA_KL_BETA="${PERSONA_KL_BETA:-0}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs
mkdir -p "$LOG"
TLOG="$LOG/train_kl${PERSONA_KL_BETA}.log"
echo "[train] starting DDv2-persona LoRA  $(date '+%F %T')  PERSONA_KL_BETA=$PERSONA_KL_BETA  SAVE_DIR=${SAVE_DIR:-<auto>}  -> $TLOG"
TRAIN_CONFIG="${TRAIN_CONFIG:-training/youdrive-ddv2-persona-lora}"
echo "[train] config=$TRAIN_CONFIG"
/root/workspace/miniconda3/envs/autovla/bin/python tools/run_sft_lora.py \
  --config "$TRAIN_CONFIG" 2>&1 | tee "$TLOG"
echo "[train] done $(date '+%F %T'). LoRA adapters -> ${SAVE_DIR:-/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/<timestamp>}"
