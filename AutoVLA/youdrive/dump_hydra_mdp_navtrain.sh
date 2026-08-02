#!/usr/bin/env bash
# Hydra-MDP dump over the navtrain12k tokens, 8 GPUs -> persona/hydra_mdp_navtrain12k.json
# Mirrors dump_hydra_mdp.sh; only the token list (-> navtrain12k), the output path, and
# SPLIT=trainval (navtrain12k tokens live in the trainval split) differ. Same ckpt/harness.
# Usage: bash dump_hydra_mdp_navtrain.sh   (override GPUs: GPUS="0 1 2 3" bash dump_hydra_mdp_navtrain.sh)
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/hydra_mdp
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export SPLIT=trainval
export TOKENS_PATH=/mnt/pfs/zhengguantian/autovla/compare/tokens_navtrain12k.json
export OUT_PATH=/mnt/pfs/zhengguantian/autovla/persona/hydra_mdp_navtrain12k.json
export CKPT=/mnt/pfs/zhengguantian/datasets/gtrs_ckpts/hydra_mdp_vov.ckpt
PY=/root/workspace/miniconda3/envs/gtrs/bin/python
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs; mkdir -p "$LOG"
pids=()
for idx in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} SHARD_IDX=$idx NUM_SHARDS=$N BATCH_SIZE=4 \
    "$PY" youdrive_dump_hydra.py > "$LOG/hydra_navtrain_shard${idx}.log" 2>&1 &
  pids+=($!); echo "hydra(navtrain) shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  echo "ERROR: one or more hydra navtrain shards failed; see $LOG/hydra_navtrain_shard*.log" >&2
  exit 1
fi
MERGE=1 "$PY" youdrive_dump_hydra.py
echo "DONE hydra_mdp navtrain -> /mnt/pfs/zhengguantian/autovla/persona/hydra_mdp_navtrain12k.json"
