#!/bin/bash
# launch_sft_full_8gpu.sh -- single-node 8-GPU full-FT SFT of the FMHead decoder
# contender (VLM LM-backbone + FMHead trained JOINTLY; codebook CE replaced by the
# FMHead flow loss). Faithful replica of tools/run_sft.py (FSDP FULL_SHARD, bf16),
# mirrors youdrive/launch_autovla_8gpu.sh env conventions. Independent config/output.
#
# Usage:
#   bash /root/workspace/fmhead/launch_sft_full_8gpu.sh
#   CONFIG=config/fmhead-sft-full.yaml bash launch_sft_full_8gpu.sh
set -u
AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY=/root/workspace/miniconda3/envs/autovla/bin/python
CONFIG="${CONFIG:-config/fmhead-sft-full.yaml}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG"
TS=$(date '+%Y-%m-%d_%H-%M-%S')
TLOG="$LOG/fmhead_sft_full_${TS}.log"

# ensure the disjoint human-GT train/val split exists (idempotent symlinks)
$PY "$FMHEAD/prepare_gt_split.py" 2>&1 | tee -a "$TLOG"

echo "[launch] FMHead full-FT SFT | config=$CONFIG | 8-GPU FSDP FULL_SHARD | log=$TLOG"
# devices='auto' in the Trainer uses all visible GPUs on the single node.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  $PY "$FMHEAD/fmhead_sft.py" --config "$FMHEAD/$CONFIG" 2>&1 | tee -a "$TLOG"
echo "[launch] done. checkpoints under /mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/<timestamp>/"
