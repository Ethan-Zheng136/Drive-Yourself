#!/usr/bin/env bash
# launch_2x8.sh -- multi-node (2 nodes x 8 GPU = 16) DDP training for FMHead.
#
# The frozen Qwen2.5-VL-3B VLM (+persona LoRA) is rebuilt per rank on its local GPU
# and runs under no_grad (OUTSIDE DDP); only the 7.36M-param FMHead decoder is
# DDP-wrapped and all-reduced. Checkpoints go to PFS; code stays in fmhead/.
#
# ------------------------------------------------------------------------------
# REQUIRED env on BOTH nodes (export before running):
#   MASTER_ADDR   IP/hostname of node 0 (rank-0 node), reachable from node 1
#   MASTER_PORT   free TCP port on MASTER_ADDR (e.g. 29500)
#   NODE_RANK     0 on the first node, 1 on the second node   <-- differs per node
# Optional:
#   NPROC         GPUs per node (default 8)
#   NNODES        number of nodes  (default 2)
#   CONFIG        fmhead config path (default config/fmhead_ddv2.yaml)
#   EXTRA         extra args forwarded to train_fmhead.py (e.g. "--max_steps 5000")
#
# Example ----------------------------------------------------------------------
#   # on node 0 (e.g. 10.0.0.1):
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 bash launch_2x8.sh
#   # on node 1:
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=1 bash launch_2x8.sh
# ------------------------------------------------------------------------------
set -euo pipefail

AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python

NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
CONFIG="${CONFIG:-config/fmhead_ddv2.yaml}"
EXTRA="${EXTRA:-}"

: "${MASTER_ADDR:?set MASTER_ADDR (rank-0 node IP)}"
: "${MASTER_PORT:?set MASTER_PORT (free TCP port on MASTER_ADDR)}"
: "${NODE_RANK:?set NODE_RANK (0 on node0, 1 on node1)}"

export PYTHONPATH="$AV:$AV/navsim:$FMHEAD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# NCCL: pick the right NIC for inter-node traffic on your cluster if needed:
#   export NCCL_SOCKET_IFNAME=eth0
#   export NCCL_IB_DISABLE=0

echo "[launch] NNODES=$NNODES NPROC=$NPROC NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "[launch] world_size = $((NNODES * NPROC))  config=$CONFIG"

"$PY" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$FMHEAD/train_fmhead.py" --config "$CONFIG" $EXTRA

# ------------------------------------------------------------------------------
# Single-node 8-GPU one-liner (no MASTER_* / NODE_RANK needed):
#   PYTHONPATH="$AV:$AV/navsim:$FMHEAD" OMP_NUM_THREADS=8 \
#     torchrun --standalone --nproc_per_node=8 \
#     /root/workspace/fmhead/train_fmhead.py --config config/fmhead_ddv2.yaml
# ------------------------------------------------------------------------------
