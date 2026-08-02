#!/usr/bin/env bash
# GTRS full dump over 4049 styletest tokens, 8 GPUs -> compare_full/gtrs.json
# Usage: bash dump_gtrs.sh   (override GPUs: GPUS="0 1 2 3" bash dump_gtrs.sh)
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
cd /root/workspace/closed_loop/GTRS
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export NAVSIM_DEVKIT_ROOT=/root/workspace/closed_loop/GTRS
export TOKENS_PATH=/mnt/pfs/zhengguantian/autovla/compare/tokens_styletest.json
export OUT_PATH=/mnt/pfs/zhengguantian/autovla/compare_full/gtrs.json
export CKPT=/mnt/pfs/zhengguantian/datasets/gtrs_ckpts/gtrs_dense_vov.ckpt
PY=/root/workspace/miniconda3/envs/gtrs/bin/python
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs; mkdir -p "$LOG"
pids=()
for idx in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} SHARD_IDX=$idx NUM_SHARDS=$N \
    "$PY" youdrive_dump_gtrs.py > "$LOG/gtrs_shard${idx}.log" 2>&1 &
  pids+=($!); echo "gtrs shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
for p in "${pids[@]}"; do wait "$p"; done
"$PY" - <<'EOF'
import json, glob
out={}; meta=None
for f in sorted(glob.glob("/mnt/pfs/zhengguantian/autovla/compare_full/gtrs.shard*of*.json")):
    d=json.load(open(f)); meta=meta or d["_meta"]
    for k,v in d.items():
        if k!="_meta": out[k]=v
meta["n_succeeded"]=len(out); meta.pop("failed",None)
final={"_meta":meta}; final.update(out)
json.dump(final, open("/mnt/pfs/zhengguantian/autovla/compare_full/gtrs.json","w"))
print("merged gtrs tokens:", len(out))
EOF
echo "DONE gtrs -> /mnt/pfs/zhengguantian/autovla/compare_full/gtrs.json"
