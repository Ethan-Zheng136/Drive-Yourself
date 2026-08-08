#!/usr/bin/env bash
# WoTE dump over the navtrain12k tokens, 8 GPUs -> persona/wote_navtrain12k.json
# Mirrors dump_wote.sh; only the token list (-> navtrain12k), the output path, and
# SPLIT=trainval (navtrain12k tokens live in the trainval split) differ. Same ckpt/harness.
# Usage: bash dump_wote_navtrain.sh   (override GPUs: GPUS="0 1 2 3" bash dump_wote_navtrain.sh)
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/WoTE
export PYTHONPATH=$PWD
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export SPLIT=trainval
export TOKENS_PATH=/mnt/pfs/zhengguantian/autovla/compare/tokens_navtrain12k.json
export OUT_PATH=/mnt/pfs/zhengguantian/autovla/persona/wote_navtrain12k.json
export CKPT=/mnt/pfs/zhengguantian/wote/ckpt/benchmark-WoTE.ckpt
PY=/root/workspace/miniconda3/envs/navsim/bin/python
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs; mkdir -p "$LOG"
pids=()
for idx in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} SHARD_IDX=$idx NUM_SHARDS=$N \
    "$PY" youdrive_dump_wote.py > "$LOG/wote_navtrain_shard${idx}.log" 2>&1 &
  pids+=($!); echo "wote(navtrain) shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  echo "ERROR: one or more wote navtrain shards failed; see $LOG/wote_navtrain_shard*.log" >&2
  exit 1
fi
MERGE=1 "$PY" youdrive_dump_wote.py
echo "DONE wote navtrain -> /mnt/pfs/zhengguantian/autovla/persona/wote_navtrain12k.json"
