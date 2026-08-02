#!/usr/bin/env bash
# GoalFlow full dump over 4049 styletest tokens, 8 GPUs -> compare_full/goalflow.json
# Usage: bash dump_goalflow.sh   (override GPUs: GPUS="0 1 2 3" bash dump_goalflow.sh)
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/GoalFlow
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export TOKENS_PATH=/mnt/pfs/zhengguantian/autovla/compare/tokens_styletest.json
export OUT_PATH=/mnt/pfs/zhengguantian/autovla/compare_full/goalflow.json
export CKPT=/mnt/pfs/zhengguantian/goalflow/ckpt/GoalFlow/goalflow_traj_epoch_54-step_18260.ckpt
export VOC_PATH=/mnt/pfs/zhengguantian/goalflow/ckpt/GoalFlow/cluster_points_8192_.npy
export SCORE_PATH=/mnt/pfs/zhengguantian/goalflow/goal_point_scores
PY=/root/workspace/miniconda3/envs/navsim/bin/python
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs; mkdir -p "$LOG"
pids=()
for idx in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} SHARD_IDX=$idx NUM_SHARDS=$N \
    "$PY" youdrive_dump_goalflow.py > "$LOG/goalflow_shard${idx}.log" 2>&1 &
  pids+=($!); echo "goalflow shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
for p in "${pids[@]}"; do wait "$p"; done
MERGE=1 "$PY" youdrive_dump_goalflow.py
echo "DONE goalflow -> /mnt/pfs/zhengguantian/autovla/compare_full/goalflow.json"
