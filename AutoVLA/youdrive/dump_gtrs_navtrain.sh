#!/usr/bin/env bash
# GTRS dump over the navtrain12k tokens, 8 GPUs -> persona/gtrs_navtrain12k.json
# Mirrors dump_gtrs.sh; only the token list (-> navtrain12k), the output path, and
# SPLIT=trainval (navtrain12k tokens live in the trainval split) differ. Same ckpt/harness.
# Usage: bash dump_gtrs_navtrain.sh   (override GPUs: GPUS="0 1 2 3" bash dump_gtrs_navtrain.sh)
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
cd /root/workspace/closed_loop/GTRS
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export NAVSIM_DEVKIT_ROOT=/root/workspace/closed_loop/GTRS
export SPLIT=trainval
export TOKENS_PATH=/mnt/pfs/zhengguantian/autovla/compare/tokens_navtrain12k.json
export OUT_PATH=/mnt/pfs/zhengguantian/autovla/persona/gtrs_navtrain12k.json
export CKPT=/mnt/pfs/zhengguantian/datasets/gtrs_ckpts/gtrs_dense_vov.ckpt
PY=/root/workspace/miniconda3/envs/gtrs/bin/python
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs; mkdir -p "$LOG"
pids=()
for idx in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} SHARD_IDX=$idx NUM_SHARDS=$N \
    "$PY" youdrive_dump_gtrs.py > "$LOG/gtrs_navtrain_shard${idx}.log" 2>&1 &
  pids+=($!); echo "gtrs(navtrain) shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  echo "ERROR: one or more gtrs navtrain shards failed; see $LOG/gtrs_navtrain_shard*.log" >&2
  exit 1
fi
# Merge the auto-suffixed per-shard files ({stem}.shardNNofMM.json) into OUT_PATH.
OUT_PATH="$OUT_PATH" "$PY" - <<'EOF'
import json, glob, os, sys
out_path = os.environ["OUT_PATH"]
stem = out_path[:-len(".json")] if out_path.endswith(".json") else out_path
parts = sorted(glob.glob(f"{stem}.shard*of*.json"))
if not parts:
    sys.exit(f"ERROR: no gtrs navtrain shard files matching {stem}.shard*of*.json")
out = {}; meta = None
for f in parts:
    d = json.load(open(f)); meta = meta or d.get("_meta")
    for k, v in d.items():
        if k != "_meta":
            out[k] = v
if meta is None:
    meta = {}
meta["n_succeeded"] = len(out); meta.pop("failed", None)
final = {"_meta": meta}; final.update(out)
json.dump(final, open(out_path, "w"))
print(f"merged gtrs navtrain tokens: {len(out)} from {len(parts)} shards -> {out_path}")
EOF
echo "DONE gtrs navtrain -> /mnt/pfs/zhengguantian/autovla/persona/gtrs_navtrain12k.json"
