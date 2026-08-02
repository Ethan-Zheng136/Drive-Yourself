#!/usr/bin/env bash
# DiffusionDriveV2 (DDv2) FULL navtest dump -> persona/ddv2_navtest.json (~12146 tokens).
#
# NON-DESTRUCTIVE: new wrapper only. It does NOT edit the DDv2 dumper
# (/root/workspace/closed_loop/DiffusionDriveV2/dump_traj.py); that driver ALREADY
# hardcodes `train_test_split=navtest`, reads TOKENS_JSON/OUT_JSON/CKPT/AGENT_CFG from
# the env, and resolves navtest logs/sensors from OPENSCENE_DATA_ROOT/{navsim_logs,sensor_blobs}/test.
# This wrapper only (a) shards the token list, (b) runs the driver once per GPU, (c) merges.
# Mirrors the proven dump_diffusiondrive_navtrain.sh sharding harness (dump_traj.py has no
# internal SHARD_IDX/NUM_SHARDS, so we shard by splitting the token file).
#
# CLUSTER-ONLY (needs 8 GPUs). Usage:
#   bash youdrive/dump_ddv2_navtest.sh                 # 8 GPUs (default)
#   GPUS="0 1 2 3" bash youdrive/dump_ddv2_navtest.sh  # subset
set -uo pipefail
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
DDV2_DIR=/root/workspace/closed_loop/DiffusionDriveV2
PY=/root/workspace/miniconda3/envs/ddv2/bin/python
DDV2_CKPT=${DDV2_CKPT:-$DDV2_DIR/ckpts/diffusiondrivev2_sel.ckpt}
AGENT_CFG=${AGENT_CFG:-diffusiondrivev2_sel_agent}
TOKENS_FULL=${TOKENS_FULL:-/mnt/pfs/zhengguantian/autovla/compare/tokens_navtest.json}
OUT_FINAL=${OUT_FINAL:-/mnt/pfs/zhengguantian/autovla/persona/ddv2_navtest.json}
SHARDS=${SHARDS:-/mnt/pfs/zhengguantian/autovla/persona/ddv2_navtest_shards}
LOG=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs
mkdir -p "$SHARDS" "$LOG"

# common navsim data env: navtest logs+sensors live under this root at .../{navsim_logs,sensor_blobs}/test
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export HF_HUB_OFFLINE=1

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
  ( cd "$DDV2_DIR" && \
    PYTHONPATH="$DDV2_DIR:$DDV2_DIR/navsim:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES=${GPUS[$idx]} \
    TOKENS_JSON="$TOK" OUT_JSON="$OUT" CKPT="$DDV2_CKPT" AGENT_CFG="$AGENT_CFG" \
    "$PY" dump_traj.py ) > "$LOG/ddv2_navtest_shard${idx}.log" 2>&1 &
  pids+=($!); echo "ddv2(navtest) shard $idx -> GPU ${GPUS[$idx]} (pid ${pids[-1]})"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  echo "ERROR: one or more ddv2 navtest shards failed; see $LOG/ddv2_navtest_shard*.log" >&2
  exit 1
fi

# merge per-shard outputs -> OUT_FINAL (keep horizon_s / n_pts / model meta from a shard).
SHARDS="$SHARDS" OUT_FINAL="$OUT_FINAL" N="$N" "$PY" - <<'EOF'
import json, os, sys
SH = os.environ["SHARDS"]; OUT = os.environ["OUT_FINAL"]; N = int(os.environ["N"])
merged = {}; meta = None; failed = []
for i in range(N):
    p = os.path.join(SH, "out_shard%d.json" % i)
    if not os.path.exists(p):
        sys.exit("ERROR: missing ddv2 navtest shard output %s" % p)
    d = json.load(open(p)); m = d.get("_meta", {})
    if meta is None:
        meta = {"model": m.get("model", "DiffusionDriveV2"),
                "horizon_s": m.get("horizon_s"), "n_pts": m.get("n_pts"),
                "frame": m.get("frame", "ego BEV, x=forward(m), y=left(m), cumulative positions"),
                "ckpt": m.get("ckpt"), "agent_cfg": m.get("agent_cfg"), "split": "navtest"}
    failed.extend(m.get("failed", []))
    for k, v in d.items():
        if k != "_meta":
            merged[k] = v
meta["n_succeeded"] = len(merged)
if failed:
    meta["failed"] = failed
out = {"_meta": meta}; out.update(merged)
json.dump(out, open(OUT, "w"))
print("[merge] ddv2 navtest tokens: %d ok, %d failed, from %d shards -> %s" % (len(merged), len(failed), N, OUT))
EOF
echo "DONE ddv2 navtest -> /mnt/pfs/zhengguantian/autovla/persona/ddv2_navtest.json"
