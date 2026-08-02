#!/usr/bin/env bash
# GRPO fine-tune of the DDv2-persona policy. Reward = PDMS (+ optional DDv2-style).
# Init + KL-reference = persona_init (styled) so KL anchors to STYLE, not base.
#
# Single 8-GPU node:   bash youdrive/train_grpo_persona.sh
# Multi-node (8x8):    the cluster launcher sets WORLD_SIZE/LOCAL_WORLD_SIZE/MASTER_ADDR via torchrun;
#                      run_rft.py reads them (num_nodes=WORLD_SIZE/LOCAL_WORLD_SIZE, devices=LOCAL_WORLD_SIZE).
#
# Optional env toggles (ablations):
#   KL_BETA=0.04           # 0 = no KL leash
#   REFERENCE_MODEL_PATH=  # default persona_init; set to base ckpt to anchor KL to base
#   PDMS_USE=1 STYLE_USE=0 STYLE_WEIGHT=1.0
set -uo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AV"
export PYTHONPATH="$AV:$AV/navsim:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_DEVKIT_ROOT="$AV/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1
# Which per-driver GRPO config to run (no .yaml suffix). Default: DDv2.
#   GRPO_CONFIG=training/youdrive-ddv2-grpo        (DDv2)
#   GRPO_CONFIG=training/youdrive-goalflow-grpo    (GoalFlow)
#   GRPO_CONFIG=training/youdrive-transfuser-grpo  (TransFuser)
GRPO_CONFIG=${GRPO_CONFIG:-training/youdrive-ddv2-grpo}
TAG=$(basename "$GRPO_CONFIG")
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG"
TLOG="$LOG/train_grpo_${TAG}.log"
# --- multi-node launch via torchrun (cluster convention) ---
# Cluster sets: WORLD_SIZE = #nodes, NPROC_PER_NODE = gpus/node, RANK = node rank, MASTER_ADDR/PORT = rendezvous.
# torchrun then exposes to the child process the *standard* env (WORLD_SIZE=total ranks,
# LOCAL_WORLD_SIZE=gpus/node), which run_rft.py reads to set num_nodes = WORLD_SIZE/LOCAL_WORLD_SIZE.
NPROC=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MADDR=${MASTER_ADDR:-127.0.0.1}
MPORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-${JOB_ID:-grpo_persona}}
RDZV_TIMEOUT=${RDZV_TIMEOUT:-1800}   # rendezvous join window (s); tolerate stragglers
echo "[grpo] start $(date '+%F %T')  nnodes=$NNODES nproc=$NPROC node_rank=$NODE_RANK rdzv=$MADDR:$MPORT id=$RDZV_ID  KL_BETA=${KL_BETA:-<cfg>} STYLE_USE=${STYLE_USE:-<cfg>}  -> $TLOG"
if [ "$NNODES" -le 1 ]; then
  # single node: static rendezvous (no inter-node DNS needed)
  /root/workspace/miniconda3/envs/autovla/bin/torchrun \
    --nproc_per_node "$NPROC" --nnodes 1 --node_rank 0 \
    --master_addr 127.0.0.1 --master_port "$MPORT" \
    tools/run_rft.py --config "$GRPO_CONFIG" 2>&1 | tee "$TLOG"
else
  # multi-node: c10d elastic rendezvous (robust to node ordering / stragglers)
  /root/workspace/miniconda3/envs/autovla/bin/torchrun \
    --nproc_per_node "$NPROC" --nnodes "$NNODES" \
    --rdzv_backend c10d --rdzv_id "$RDZV_ID" --rdzv_endpoint "$MADDR:$MPORT" \
    --rdzv_conf "timeout=$RDZV_TIMEOUT" \
    tools/run_rft.py --config "$GRPO_CONFIG" 2>&1 | tee "$TLOG"
fi
echo "[grpo] done $(date '+%F %T'). ckpts -> $AV/runs/grpo/"
