#!/usr/bin/env bash
# DiffusionDrive (v1) dump over the navtrain12k tokens, 8 GPUs -> persona/diffusiondrive_navtrain12k.json
#
# DiffusionDrive (v1) has no standalone dump_*.sh: its proven harness is the
# run_diffusiondrive() path inside dump_all_full.sh, which uses the generic 8-GPU
# `launch` mechanism (split the token list into N per-shard token files, run
# diffusiondrive/dump_traj.py once per GPU, then merge). dump_traj.py has no
# internal SHARD_IDX/NUM_SHARDS, so sharding is done here by token-file splitting,
# mirroring dump_all_full.sh exactly. Only the token list (-> navtrain12k), the
# output path, and SPLIT=trainval (navtrain12k tokens live in the trainval split)
# differ from the styletest run; same ckpt (DD_CKPT) / python env / harness.
#
# Usage: bash dump_diffusiondrive_navtrain.sh   (override GPUs: GPUS="0 1 2 3" bash dump_diffusiondrive_navtrain.sh)
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
DD_DIR=/root/workspace/closed_loop/diffusiondrive
PY=/root/workspace/miniconda3/envs/diffusiondrive/bin/python
DD_CKPT=/mnt/pfs/zhengguantian/output/g_csa_vad/ckpts/diffusiondrive/diffusiondrive_navsim_88p1_PDMS
TOKENS_FULL=/mnt/pfs/zhengguantian/autovla/compare/tokens_navtrain12k.json
OUT_FINAL=/mnt/pfs/zhengguantian/autovla/persona/diffusiondrive_navtrain12k.json
SHARDS=/mnt/pfs/zhengguantian/autovla/persona/diffusiondrive_navtrain_shards
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs
mkdir -p "$SHARDS" "$LOG"

# common navsim data env (mirrors dump_all_full.sh)
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export HF_HUB_OFFLINE=1
export SPLIT=trainval

# split the full token list into N round-robin shards (one token file per GPU).
TOKENS_FULL="$TOKENS_FULL" SHARDS="$SHARDS" N="$N" "$PY" - <<'EOF'
import json, os
toks = json.load(open(os.environ["TOKENS_FULL"]))
N = int(os.environ["N"]); SH = os.environ["SHARDS"]
for i in range(N):
    json.dump(toks[i::N], open(os.path.join(SH, "tokens_shard%d.json" % i), "w"))
print("[split] %d tokens -> %d shards: %s" % (len(toks), N, [len(toks[i::N]) for i in range(N)]))
EOF

pids=()
for idx in "${!GPUS[@]}"; do
  TOK="$SHARDS/tokens_shard${idx}.json"
  OUT="$SHARDS/out_shard${idx}.json"
  ( cd "$DD_DIR" && \
    PYTHONPATH="$DD_DIR:$DD_DIR/navsim:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES=${GPUS[$idx]} \
    TOKENS_PATH="$TOK" OUT_PATH="$OUT" CKPT="$DD_CKPT" \
    "$PY" dump_traj.py ) > "$LOG/diffusiondrive_navtrain_shard${idx}.log" 2>&1 &
  pids+=($!); echo "diffusiondrive(navtrain) shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  echo "ERROR: one or more diffusiondrive navtrain shards failed; see $LOG/diffusiondrive_navtrain_shard*.log" >&2
  exit 1
fi

# merge per-shard outputs -> OUT_FINAL (keep horizon_s / n_pts from the shard _meta).
SHARDS="$SHARDS" OUT_FINAL="$OUT_FINAL" N="$N" "$PY" - <<'EOF'
import json, os, sys
SH = os.environ["SHARDS"]; OUT = os.environ["OUT_FINAL"]; N = int(os.environ["N"])
merged = {}; meta = None
for i in range(N):
    p = os.path.join(SH, "out_shard%d.json" % i)
    if not os.path.exists(p):
        sys.exit("ERROR: missing diffusiondrive navtrain shard output %s" % p)
    d = json.load(open(p)); m = d.get("_meta", {})
    if meta is None:
        meta = {"model": m.get("model", "DiffusionDrive"),
                "horizon_s": m.get("horizon_s"), "n_pts": m.get("n_pts"),
                "frame": m.get("frame", "ego BEV, x=forward(m), y=left(m), cumulative positions"),
                "ckpt": m.get("ckpt")}
    for k, v in d.items():
        if k != "_meta":
            merged[k] = v
meta["n_succeeded"] = len(merged)
out = {"_meta": meta}; out.update(merged)
json.dump(out, open(OUT, "w"))
print("[merge] diffusiondrive navtrain tokens: %d from %d shards -> %s" % (len(merged), N, OUT))
EOF
echo "DONE diffusiondrive navtrain -> /mnt/pfs/zhengguantian/autovla/persona/diffusiondrive_navtrain12k.json"
