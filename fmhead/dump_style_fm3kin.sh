#!/usr/bin/env bash
# dump_style_fm3kin.sh -- FULL 13-dim style-metric profile (5 kin + 5 social + 3 YDSP) for a
# fm3-kin FMHead base at ONE epoch, STYLE OFF (s=0, no persona LoRA -> feasibility base's own
# driving profile). Chains: (1) 8-GPU sharded FMHead dump (KINEMATIC decode) -> (2) merge shards
# -> (3) youdrive/sweep_metrics.py -> style_metrics.{json,txt,png} in OUTDIR.
#
# fm3-kin MUST decode in kinematic mode: control normalizer traj_norm_stats_ctrl.json +
# FMHEAD_CONSTRAINT (unicycle caps) + v0 from vehicle_velocity + steps=100. Without it the
# (a_long,yaw_rate) control samples get read as raw xy -> garbage.
#
# Usage:  EP=6  bash /root/workspace/fmhead/dump_style_fm3kin.sh
#         EP=10 bash /root/workspace/fmhead/dump_style_fm3kin.sh
#         EP=14 bash /root/workspace/fmhead/dump_style_fm3kin.sh
# GPUS defaults to all 8; single-GPU node -> GPUS="0" bash ... (auto single-shard).
set -uo pipefail

EP="${EP:?set EP=6|10|14}"
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
KIN_DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42
PY=/root/workspace/miniconda3/envs/autovla/bin/python
cd "$AV"

# ---- ckpt names: ep14 is the run "final" (unsuffixed); ep6/ep10 are extracted epoch snaps ----
case "$EP" in
  6)  BASE_SRC="$KIN_DIR/fullft_fm3_kin_ep6_base.ckpt";  HEAD="$KIN_DIR/fullft_fm3_kin_ep6_head.pt"  ;;
  10) BASE_SRC="$KIN_DIR/fullft_fm3_kin_ep10_base.ckpt"; HEAD="$KIN_DIR/fullft_fm3_kin_ep10_head.pt" ;;
  14) BASE_SRC="$KIN_DIR/fullft_fm3_kin_base.ckpt";      HEAD="$KIN_DIR/fullft_fm3_kin_head.pt"      ;;
  *)  echo "[style] EP must be 6|10|14 (got $EP)"; exit 2 ;;
esac
[ -f "$BASE_SRC" ] || { echo "[style] missing base $BASE_SRC"; exit 2; }
[ -f "$HEAD" ]     || { echo "[style] missing head $HEAD"; exit 2; }

OUTDIR="/mnt/pfs/zhengguantian/autovla/persona/style_fm3kin_ep${EP}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$OUTDIR" "$LOG"

# ---- navsim/nuplan data-root env (needed by sweep_metrics' env_interact social pass) ----
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AV/navsim"
export PYTHONPATH="$AV:$AV/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
unset PERSONA_ADAPTERS   # STYLE OFF (s=0): base's own profile, no persona LoRA

# ---- copy 8GB base ckpt to node-local once (avoid PFS I/O storm across shards) ----
LOCAL_BASE="/dev/shm/fm3kin_ep${EP}_base.ckpt"
if [ ! -f "$LOCAL_BASE" ]; then
  echo "[style] copying base -> $LOCAL_BASE ..."
  cp "$BASE_SRC" "$LOCAL_BASE" 2>/dev/null || { LOCAL_BASE="/tmp/fm3kin_ep${EP}_base.ckpt"; cp "$BASE_SRC" "$LOCAL_BASE"; }
fi

# ---- SELECT=pdm (DEPLOYED 0.9088 selection): score N cands with PDM_Reward, argmax. Needs the
# navtest metric cache -> copy to node-local once so all shards read local (same as run_pdms). ----
SELECT="${SELECT:-pdm}"
SRC_CACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
CACHE="${LOCAL_CACHE:-/tmp/metric_cache_navtest_v1}"
if [ "$SELECT" = "pdm" ] && [ ! -d "$CACHE/metadata" ]; then
  echo "[style] copying metric cache -> $CACHE ..."; rm -rf "$CACHE"; cp -r "$SRC_CACHE" "$CACHE"
fi

GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
echo "[style] EP=$EP  base=$BASE_SRC  head=$(basename "$HEAD")  select=$SELECT  gpus=${GPUS[*]}  outdir=$OUTDIR"

# ---- (1) sharded KINEMATIC dump (style off, control norm, unicycle constraint, steps=100) ----
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" SHARD_IDX="$i" NUM_SHARDS="$N" \
    BASE_CKPT="$LOCAL_BASE" \
    FMHEAD_CKPT="$HEAD" \
    FMHEAD_NORM="$FMHEAD/traj_norm_stats_ctrl.json" \
    FMHEAD_CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}' \
    FMHEAD_STEPS=100 FMHEAD_N=16 SELECT="$SELECT" METRIC_CACHE="$CACHE" ALPHAS=0 \
    OUTDIR="$OUTDIR" \
    "$PY" "$FMHEAD/dump_fmhead_style.py" > "$LOG/style_fm3kin_ep${EP}_shard$i.log" 2>&1 &
  pids+=($!); echo "  shard $i -> GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
FAIL=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  shard $i OK"; else echo "  shard $i FAILED (see $LOG/style_fm3kin_ep${EP}_shard$i.log)"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || { echo "[style] a shard failed -- aborting before merge"; exit 4; }

# ---- (2) merge shards -> persona_a0.json (single-shard run already writes it directly) ----
if [ "$N" -gt 1 ]; then
  "$PY" - "$OUTDIR" <<'PYEOF'
import json, glob, os, sys
outdir = sys.argv[1]
merged, meta = {}, None
for f in sorted(glob.glob(os.path.join(outdir, "persona_a0.shard*.json"))):
    d = json.load(open(f)); meta = meta or d.get("_meta")
    for k, v in d.items():
        if k != "_meta": merged[k] = v
out = {"_meta": meta} if meta else {}; out.update(merged)
json.dump(out, open(os.path.join(outdir, "persona_a0.json"), "w"))
print(f"[style] merged {len(merged)} tokens -> persona_a0.json")
PYEOF
fi

# ---- (3) full 13-dim profile: 3 YDSP + 5 kin + 5 social -> style_metrics.{json,txt,png} ----
"$PY" "$AV/youdrive/sweep_metrics.py" --outdir "$OUTDIR"
echo "[style] DONE ep$EP -> $OUTDIR/style_metrics.txt"
