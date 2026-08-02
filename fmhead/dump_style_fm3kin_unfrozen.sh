#!/usr/bin/env bash
# dump_style_fm3kin_unfrozen.sh -- STYLE dose-response alpha-sweep for the UNFROZEN-head
# style ckpt on the fm3-kin (kinematic, control-space) base. Chains: (1) N-GPU sharded
# FMHead dump with the DDv2 persona LoRA attached (style axis ON) over ALPHAS -> (2) merge
# shards per alpha -> (3) youdrive/sweep_metrics.py -> style_metrics.{json,txt,png} in OUTDIR.
#
# vs dump_style_fm3kin.sh (style OFF, s=0): this attaches PERSONA_ADAPTERS (ddv2 LoRA) and
# sweeps ALPHAS with FMHEAD_USE_STYLE_ADAPTER=true so we get the L2->DDv2 / kin / social
# dose-response of the UNFROZEN head vs the DDv2 teacher.
#
# Usage:
#   bash dump_style_fm3kin_unfrozen.sh                                   # full 8-GPU sweep
#   GPUS="0" ALPHAS="0,1" TOKENS=/path/smoke10.json OUTDIR=/mnt/.../smoke SKIP_SWEEP=0 \
#       bash dump_style_fm3kin_unfrozen.sh                              # smoke
set -uo pipefail

AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
cd "$AV"

# ---- trained artifacts (unfrozen head + fm3-kin ep6 base + DDv2 persona LoRA) ----
HEAD="${FMHEAD_CKPT:-/mnt/pfs/zhengguantian/autovla/persona/fmhead_style_ddv2_fm3kin_unfrozen/2026-07-18_01-47-36_x8/fmhead_final.pt}"
BASE_SRC="${BASE_CKPT:-/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42/fullft_fm3_kin_ep6_base.ckpt}"
LORA="${PERSONA_ADAPTERS:-/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/ddv2_big_final/lora_final}"
NORM="$FMHEAD/traj_norm_stats_ctrl.json"
CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"
SELECT="${SELECT:-pdm}"
OUTDIR="${OUTDIR:-/mnt/pfs/zhengguantian/autovla/persona/style_fm3kin_unfrozen_sweep}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$OUTDIR" "$LOG"
[ -f "$HEAD" ]     || { echo "[unfroz] missing head $HEAD"; exit 2; }
[ -f "$BASE_SRC" ] || { echo "[unfroz] missing base $BASE_SRC"; exit 2; }
[ -d "$LORA" ]     || { echo "[unfroz] missing lora $LORA"; exit 2; }

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AV/navsim"
export PYTHONPATH="$AV:$AV/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1

# ---- base ckpt -> node-local once (avoid PFS I/O storm across shards) ----
LOCAL_BASE="${LOCAL_BASE:-/dev/shm/fm3kin_ep6_base.ckpt}"
if [ ! -f "$LOCAL_BASE" ]; then
  echo "[unfroz] copying base -> $LOCAL_BASE ..."
  cp "$BASE_SRC" "$LOCAL_BASE" 2>/dev/null || { LOCAL_BASE="/tmp/fm3kin_ep6_base.ckpt"; cp "$BASE_SRC" "$LOCAL_BASE"; }
fi

# ---- SELECT=pdm needs the navtest metric cache; copy node-local once for the full run ----
SRC_CACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
CACHE="${METRIC_CACHE:-/tmp/metric_cache_navtest_v1}"
if [ "$SELECT" = "pdm" ] && [ ! -d "$CACHE/metadata" ]; then
  echo "[unfroz] copying metric cache -> $CACHE ..."; rm -rf "$CACHE"; cp -r "$SRC_CACHE" "$CACHE"
fi

GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
echo "[unfroz] head=$(basename "$HEAD") base=$BASE_SRC lora=$(basename "$LORA")"
echo "[unfroz] alphas=$ALPHAS select=$SELECT gpus=${GPUS[*]} outdir=$OUTDIR tokens=${TOKENS:-ALL}"

# ---- (1) sharded KINEMATIC dump, persona LoRA ON, ALPHAS sweep ----
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" SHARD_IDX="$i" NUM_SHARDS="$N" \
    BASE_CKPT="$LOCAL_BASE" FMHEAD_CKPT="$HEAD" FMHEAD_NORM="$NORM" \
    PERSONA_ADAPTERS="$LORA" FMHEAD_USE_STYLE_ADAPTER=true \
    FMHEAD_CONSTRAINT="$CONSTRAINT" FMHEAD_STEPS=100 FMHEAD_N=16 \
    SELECT="$SELECT" METRIC_CACHE="$CACHE" ALPHAS="$ALPHAS" OUTDIR="$OUTDIR" \
    TOKENS="${TOKENS:-}" \
    "$PY" "$FMHEAD/dump_fmhead_style.py" > "$LOG/style_unfrozen_shard$i.log" 2>&1 &
  pids+=($!); echo "  shard $i -> GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
FAIL=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  shard $i OK"; else echo "  shard $i FAILED (see $LOG/style_unfrozen_shard$i.log)"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || { echo "[unfroz] a shard failed -- aborting before merge"; exit 4; }

# ---- (2) merge shards per alpha (single-shard writes persona_a{a}.json directly) ----
if [ "$N" -gt 1 ]; then
  "$PY" - "$OUTDIR" "$ALPHAS" <<'PYEOF'
import json, glob, os, sys
outdir, alphas = sys.argv[1], sys.argv[2].split(",")
for a in alphas:
    a = float(a); merged, meta = {}, None
    tag = f"{a:g}"
    for f in sorted(glob.glob(os.path.join(outdir, f"persona_a{tag}.shard*.json"))):
        d = json.load(open(f)); meta = meta or d.get("_meta")
        for k, v in d.items():
            if k != "_meta": merged[k] = v
    out = {"_meta": meta} if meta else {}; out.update(merged)
    json.dump(out, open(os.path.join(outdir, f"persona_a{tag}.json"), "w"))
    print(f"[unfroz] merged alpha={tag}: {len(merged)} tokens -> persona_a{tag}.json")
PYEOF
fi

# ---- (3) full 13-dim style profile (YDSP + kin + social) + dose-response plot ----
if [ "${SKIP_SWEEP:-0}" != "1" ]; then
  "$PY" "$AV/youdrive/sweep_metrics.py" --outdir "$OUTDIR"
  echo "[unfroz] DONE -> $OUTDIR/style_metrics.txt"
fi
