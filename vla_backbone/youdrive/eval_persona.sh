#!/usr/bin/env bash
# No-merge persona EPDMS eval: base AutoVLA + attached LoRA vector(s) at given weights.
# Mirrors launch_autovla_8gpu.sh but injects PERSONA_ADAPTERS/PERSONA_WEIGHTS (agent attaches on top).
# Usage:
#   ADAPTERS=/path/lora_final WEIGHTS=1.0 TAG=ddv2_a1 bash youdrive/eval_persona.sh
#   ADAPTERS=v_a,v_b WEIGHTS=0.5,-0.5 TAG=mix bash youdrive/eval_persona.sh   # task arithmetic
#   SMOKE=1 ADAPTERS=/path/lora_final WEIGHTS=1.0 TAG=smk bash youdrive/eval_persona.sh   # ~20 tokens, 1 GPU
set -uo pipefail
AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
# ---- the no-merge knob: agent reads these and attaches adapters on top of base ----
export PERSONA_ADAPTERS="${ADAPTERS:?set ADAPTERS=comma-sep adapter dirs}"
export PERSONA_WEIGHTS="${WEIGHTS:-1.0}"

PY=/root/workspace/miniconda3/envs/autovla/bin/python
export PYTHONUNBUFFERED=1   # live logs (see "Processing scenario" immediately)
# avoid pfs I/O storm: copy 16GB base AND 3GB metric_cache to node-local; all shards read local
SRC_BASE=/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt
SRC_CACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
CKPT="${LOCAL_BASE:-/dev/shm/autovla_base.ckpt}"
if [ ! -f "$CKPT" ]; then
  echo "[eval_persona] copying base -> $CKPT ..."
  cp "$SRC_BASE" "$CKPT" 2>/dev/null || { CKPT=/tmp/autovla_base.ckpt; cp "$SRC_BASE" "$CKPT"; }
fi
CACHE="${LOCAL_CACHE:-/tmp/metric_cache_navtest_v1}"
if [ ! -d "$CACHE/metadata" ]; then
  echo "[eval_persona] copying metric_cache -> $CACHE ..."
  rm -rf "$CACHE"; cp -r "$SRC_CACHE" "$CACHE"
fi
CFG=config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml
SF_DIR=navsim/navsim/planning/script/config/common/train_test_split/scene_filter
TAG="${TAG:-persona}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
echo "[eval_persona] adapters=$PERSONA_ADAPTERS weights=$PERSONA_WEIGHTS tag=$TAG gpus=${GPUS[*]}"

# ---- build shard scene_filters (SMOKE: 20 tokens on 1 shard) ----
if [ "${SMOKE:-0}" = "1" ]; then GPUS=(0); NSHARD=1; NTOK=20; else NSHARD=${#GPUS[@]}; NTOK=0; fi
echo "===== [1/3] generating $NSHARD token shards $(date '+%T') ====="
$PY - "$SF_DIR" "$NSHARD" "$NTOK" "$TAG" <<'PYEOF'
import sys, yaml, copy
sf_dir, nshard, ntok, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = yaml.safe_load(open(f"{sf_dir}/navtest.yaml")); toks = cfg.get("tokens") or []
if ntok > 0: toks = toks[:ntok]
for i in range(nshard):
    c = copy.deepcopy(cfg); c["tokens"] = toks[i::nshard]
    yaml.safe_dump(c, open(f"{sf_dir}/persona_{tag}_shard{i}.yaml", "w"))
    print(f"  shard{i}: {len(c['tokens'])} tokens")
PYEOF

# ---- run shards (inner & + wait = safe; outer script stays foreground) ----
echo "===== [2/3] launching $NSHARD GPU workers $(date '+%T') ====="
pids=()
for i in $(seq 0 $((NSHARD-1))); do
  gpu=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$gpu $PY navsim/navsim/planning/script/run_pdm_score_cot.py \
    train_test_split=navtest \
    train_test_split/scene_filter=persona_${TAG}_shard$i \
    agent=autovla_agent +agent.config_path=$CFG +agent.checkpoint_path=$CKPT \
    +agent.sensor_data_path=. +agent.lora_conf.use_lora=false \
    metric_cache_path=$CACHE json_data_path=dataset/nuplan/navtest_nocot \
    experiment_name=persona_${TAG}_shard$i > "$LOG/eval_${TAG}_shard$i.log" 2>&1 &
  pids+=($!); echo "  shard $i -> GPU $gpu, pid ${pids[-1]}"
done

echo "===== waiting for all shards ====="
for i in $(seq 0 $((NSHARD-1))); do
  if wait "${pids[$i]}"; then echo "  shard $i OK"; else echo "  shard $i FAILED"; fi
done

# ---- aggregate PDMS (separate .py to avoid bash/heredoc parsing of python) ----
echo "===== [3/3] merging + scoring $(date '+%T') ====="
$PY "$(dirname "$0")/_agg_pdms.py" "$TAG" "$NSHARD"
echo "ALL_DONE $(date '+%T')"
