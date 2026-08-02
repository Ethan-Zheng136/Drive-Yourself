#!/usr/bin/env bash
# launch_lora_fm3aligned_8gpu.sh -- single-node 8-GPU DDP training for the
# "fm3-aligned persona LoRA" recipe (torchrun standalone).
#
# Usage:
#   CONFIG=config/fmhead_lora_fm3aligned_ddv2.yaml bash launch_lora_fm3aligned_8gpu.sh
#   NPROC=8 CONFIG=... EXTRA="--max_steps 12000" bash launch_lora_fm3aligned_8gpu.sh
#
# FROZEN fm3-kin ep6 base VLM + FROZEN fm3-kin ep6 head per rank on cuda:LOCAL_RANK; only the
# FRESH persona LoRA (r=64/alpha=128, attn+MLP) is trainable and DDP-all-reduced. Ckpts -> PFS.
set -euo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python

NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-config/fmhead_lora_fm3aligned_ddv2.yaml}"
EXTRA="${EXTRA:-}"

export PYTHONPATH="$AV:$AV/navsim:$FMHEAD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[launch_lora_fm3aligned_8gpu] nproc=$NPROC config=$CONFIG world_size=$NPROC"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$FMHEAD/train_lora_fm3aligned.py" --config "$CONFIG" $EXTRA
