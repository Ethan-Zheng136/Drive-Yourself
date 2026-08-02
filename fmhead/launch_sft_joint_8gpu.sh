#!/bin/bash
# launch_sft_joint_8gpu.sh -- single-node 8-GPU JOINT training: the fm3-kin full-FT SFT
# recipe PLUS a jointly-fired FMHeadScorer distilled from in-loop pdm_score (see
# fmhead_sft_joint.py). Same FSDP FULL_SHARD / bf16 setup as launch_sft_full_8gpu.sh; the
# ONLY differences are the entrypoint (fmhead_sft_joint.py) and the default config
# (config/fmhead_fm3_kin_joint.yaml). The navsim env vars below are REQUIRED because the
# scorer's per-candidate labels come from the real NAVSIM pdm_score (needs maps + cache).
#
# Usage:
#   bash /root/workspace/fmhead/launch_sft_joint_8gpu.sh
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
CONFIG="${CONFIG:-config/fmhead_fm3_kin_joint.yaml}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG"
TS=$(date '+%Y-%m-%d_%H-%M-%S')
TLOG="$LOG/fmhead_sft_joint_${TS}.log"

# ensure the disjoint human-GT navtrain train/val split exists (idempotent symlinks)
$PY "$FMHEAD/prepare_gt_split.py" 2>&1 | tee -a "$TLOG"

echo "[launch] FMHead JOINT SFT (base+head+scorer) | config=$CONFIG | 8-GPU FSDP | log=$TLOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  $PY "$FMHEAD/fmhead_sft_joint.py" --config "$FMHEAD/$CONFIG" 2>&1 | tee -a "$TLOG"
echo "[launch] done. ckpts under /mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin_joint/<ts>/"
echo "[launch] extract deployable pieces (base/head/scorer) with:"
echo "  base+head: run_fm3_extract.sh-style; scorer: extract_scorer.py <epoch=*.ckpt> <out.pt>"
