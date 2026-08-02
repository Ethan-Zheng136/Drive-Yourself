#!/usr/bin/env bash
# launch_8gpu.sh -- single-node 8-GPU DDP training for FMHead (torchrun standalone).
#
# Usage:
#   CONFIG=config/fmhead_gt_feasibility.yaml bash launch_8gpu.sh
#   NPROC=8 CONFIG=... EXTRA="--max_steps 20000" bash launch_8gpu.sh
#
# Frozen Qwen2.5-VL-3B(+optional LoRA) per rank on cuda:LOCAL_RANK (no_grad, OUTSIDE
# DDP); only the 7.36M FMHead is DDP-wrapped/all-reduced. Checkpoints -> PFS.
set -euo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python

NPROC="${NPROC:-8}"
CONFIG="${CONFIG:-config/fmhead_gt_feasibility.yaml}"
EXTRA="${EXTRA:-}"

export PYTHONPATH="$AV:$AV/navsim:$FMHEAD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[launch_8gpu] nproc=$NPROC config=$CONFIG world_size=$NPROC"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$FMHEAD/train_fmhead.py" --config "$CONFIG" $EXTRA
