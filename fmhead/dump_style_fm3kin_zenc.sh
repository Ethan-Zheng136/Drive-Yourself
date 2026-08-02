#!/usr/bin/env bash
# dump_style_fm3kin_zenc.sh -- STYLE dose-response alpha-sweep for the Z-ENCODER style ckpt on
# the fm3-kin (kinematic, control-space) base. Same chain as dump_style_fm3kin_unfrozen.sh but
# for the z-path head: NO style adapter (FMHEAD_USE_Z_ENCODER=true suppresses the adapter
# heuristic; the agent auto-detects the z_encoder from the ckpt keys), content from the LoRA-OFF
# forward (FMHEAD_CONTENT_SOURCE=base -> neutral body). Sweeps ALPHAS (and optionally CFG_W).
#
# Chains: (1) N-GPU sharded FMHead dump, DDv2 persona LoRA ON, ALPHAS sweep, PDM select ->
#         (2) merge shards per alpha -> (3) youdrive/sweep_metrics.py -> style_metrics.{json,txt,png}.
#
# Usage:
#   FMHEAD_CKPT=/mnt/pfs/.../fmhead_style_ddv2_zenc/<run>/fmhead_final.pt bash dump_style_fm3kin_zenc.sh
#   GPUS="0" ALPHAS="0,1" TOKENS=/path/smoke10.json OUTDIR=/mnt/.../smoke SKIP_SWEEP=1 \
#       FMHEAD_CKPT=... bash dump_style_fm3kin_zenc.sh                                  # smoke
set -uo pipefail

AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
cd "$AV"

HEAD="${FMHEAD_CKPT:?set FMHEAD_CKPT=/mnt/pfs/.../fmhead_style_ddv2_zenc/<run>/fmhead_final.pt}"
BASE_SRC="${BASE_CKPT:-/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42/fullft_fm3_kin_ep6_base.ckpt}"
LORA="${PERSONA_ADAPTERS:-/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/ddv2_big_final/lora_final}"
NORM="$FMHEAD/traj_norm_stats_ctrl.json"
CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"
CFG_W="${CFG_W:-1.0}"
SELECT="${SELECT:-pdm}"
OUTDIR="${OUTDIR:-/mnt/pfs/zhengguantian/autovla/persona/style_fm3kin_zenc_sweep}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$OUTDIR" "$LOG"
[ -f "$HEAD" ]     || { echo "[zenc] missing head $HEAD"; exit 2; }
[ -f "$BASE_SRC" ] || { echo "[zenc] missing base $BASE_SRC"; exit 2; }
[ -d "$LORA" ]     || { echo "[zenc] missing lora $LORA"; exit 2; }

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
  echo "[zenc] copying base -> $LOCAL_BASE ..."
  cp "$BASE_SRC" "$LOCAL_BASE" 2>/dev/null || { LOCAL_BASE="/tmp/fm3kin_ep6_base.ckpt"; cp "$BASE_SRC" "$LOCAL_BASE"; }
fi

# ---- PDM metric cache node-local (reuse the SAME fixed pdm+v0 cache as every other run) ----
SRC_CACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
CACHE="${METRIC_CACHE:-/tmp/metric_cache_navtest_v1}"
if [ "$SELECT" = "pdm" ] && [ ! -d "$CACHE/metadata" ]; then
  echo "[zenc] copying metric cache -> $CACHE ..."; rm -rf "$CACHE"; cp -r "$SRC_CACHE" "$CACHE"
fi

GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
echo "[zenc] head=$(basename "$HEAD") base=$BASE_SRC lora=$(basename "$LORA")"
echo "[zenc] alphas=$ALPHAS cfg_w=$CFG_W select=$SELECT gpus=${GPUS[*]} outdir=$OUTDIR tokens=${TOKENS:-ALL}"

# ---- (1) sharded KINEMATIC dump, persona LoRA ON, Z-ENCODER path, base content, ALPHAS sweep ----
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" SHARD_IDX="$i" NUM_SHARDS="$N" \
    BASE_CKPT="$LOCAL_BASE" FMHEAD_CKPT="$HEAD" FMHEAD_NORM="$NORM" \
    PERSONA_ADAPTERS="$LORA" FMHEAD_USE_Z_ENCODER=true FMHEAD_CONTENT_SOURCE=base \
    FMHEAD_CONSTRAINT="$CONSTRAINT" FMHEAD_STEPS=100 FMHEAD_N=16 FMHEAD_CFG_WEIGHT="$CFG_W" \
    SELECT="$SELECT" METRIC_CACHE="$CACHE" ALPHAS="$ALPHAS" OUTDIR="$OUTDIR" \
    TOKENS="${TOKENS:-}" \
    "$PY" "$FMHEAD/dump_fmhead_style.py" > "$LOG/style_zenc_shard$i.log" 2>&1 &
  pids+=($!); echo "  shard $i -> GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
FAIL=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  shard $i OK"; else echo "  shard $i FAILED (see $LOG/style_zenc_shard$i.log)"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || { echo "[zenc] a shard failed -- aborting before merge"; exit 4; }

# ---- (2) merge shards per alpha ----
if [ "$N" -gt 1 ]; then
  "$PY" - "$OUTDIR" "$ALPHAS" <<'PYEOF'
import json, glob, os, sys
outdir, alphas = sys.argv[1], sys.argv[2].split(",")
for a in alphas:
    a = float(a); merged, meta = {}, None; tag = f"{a:g}"
    for f in sorted(glob.glob(os.path.join(outdir, f"persona_a{tag}.shard*.json"))):
        d = json.load(open(f)); meta = meta or d.get("_meta")
        for k, v in d.items():
            if k != "_meta": merged[k] = v
    out = {"_meta": meta} if meta else {}; out.update(merged)
    json.dump(out, open(os.path.join(outdir, f"persona_a{tag}.json"), "w"))
    print(f"[zenc] merged alpha={tag}: {len(merged)} tokens -> persona_a{tag}.json")
PYEOF
fi

# ---- (3) full 13-dim style profile + dose-response plot ----
if [ "${SKIP_SWEEP:-0}" != "1" ]; then
  "$PY" "$AV/youdrive/sweep_metrics.py" --outdir "$OUTDIR"
  echo "[zenc] DONE -> $OUTDIR/style_metrics.txt"
fi
