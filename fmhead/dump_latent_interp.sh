#!/usr/bin/env bash
# dump_latent_interp.sh -- INFERENCE-ONLY "latent content interpolation" style alpha-sweep on the
# FROZEN fm3-kin (kinematic, control-space) ep6 base + head. NO training, NO new trainable module.
#
# Mechanism (content_source=interp): feed the FMHead's cross-attn CONTENT as the alpha-interpolated
# VLM latent  c(alpha) = ctx_base + alpha*(ctx_styled - ctx_base), with the AdaLN style s=0 (NO
# z-encoder, NO adapter, NO CFG style path). alpha is the ONLY knob:
#   alpha=0   -> c=ctx_base -> EXACTLY the clean fm3-kin base (bit-exact neutral),
#   alpha up  -> content latent shifts toward the DDv2 persona -> style influences the whole decode.
# ctx_base = LoRA-OFF forward, ctx_styled = DDv2 persona LoRA-ON forward.
#
# Chain: (1) 1-GPU (or sharded) FMHead dump, DDv2 persona LoRA ON, ALPHAS sweep, content_source=interp,
#        kinematic constraint + control normalizer + v0, PDM select -> (2) merge shards per alpha ->
#        (3) youdrive/sweep_metrics.py -> style_metrics.{json,txt,png} -> (4) style_metrics summary.
#
# Usage (single GPU, LIMIT subset for speed -- the intended vision sweep):
#   GPUS="0" LIMIT=400 bash dump_latent_interp.sh
# Full styletest set (multi-GPU sharded):
#   GPUS="0 1 2 3 4 5 6 7" bash dump_latent_interp.sh
# Env knobs: ALPHAS, LIMIT, GPUS, OUTDIR, SELECT, SKIP_SWEEP, FMHEAD_CKPT, BASE_CKPT, PERSONA_ADAPTERS,
#            PERSONA_WEIGHTS.
#
# Integrated navtest PDMS/safety (ADDITIVE, default ON): after the style dump+sweep, this script
# ALSO runs run_pdms_fmhead.sh in interp mode per alpha (same persona adapters/weights + kinematic
# constraint + control normalizer + FMHEAD_CONTENT_SOURCE=interp) so ONE command yields BOTH the
# style dose-response AND the PDMS numbers. Collected into <OUTDIR>/pdms_summary.txt (alpha -> PDMS).
#   RUN_PDMS=1 (default) run PDMS ; RUN_PDMS=0 skip -> byte-identical to the old style-only path.
#   PDMS_ALPHAS="0,1.0"  optional subset of ALPHAS for PDMS only (default = ALL of ALPHAS).
# PDMS is navtest (a SEPARATE token set from styletest) -> a separate decode; that is expected.
#
# Multi-adapter task arithmetic (INFERENCE-ONLY, no training): PERSONA_ADAPTERS may be a
# comma-separated list of persona LoRA dirs and PERSONA_WEIGHTS a matching comma-separated list of
# floats. The base AutoVLAAgent.initialize attaches all adapters and activates them together, so the
# LoRA-ON forward is the weighted stack (PEFT applies active adapters ADDITIVELY, per-adapter scaling
# = its weight). The interp content is then c(alpha) = ctx_base + alpha*(ctx_styled_combined -
# ctx_base), i.e. c = h_base + alpha*(w1*delta_1 + w2*delta_2 + ...). Example:
#   PERSONA_ADAPTERS="/path/ddv2/lora_final,/path/transfuser/lora_final" PERSONA_WEIGHTS="0.5,0.5" \
#     ALPHAS="0,0.5,1.0" LIMIT=400 OUTDIR=/mnt/pfs/.../style_latent_interp_ddv2_transfuser \
#     GPUS="0" bash dump_latent_interp.sh
# PERSONA_WEIGHTS defaults to EQUAL weights (1/n) so a single adapter stays at 1.0 (byte-identical).
set -uo pipefail

AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
cd "$AV"

# ---- frozen fm3-kin ep6 base + head (GOLD/protected -- never modified) + DDv2 persona LoRA ----
FM3DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42
HEAD="${FMHEAD_CKPT:-$FM3DIR/fullft_fm3_kin_ep6_head.pt}"
BASE_SRC="${BASE_CKPT:-$FM3DIR/fullft_fm3_kin_ep6_base.ckpt}"
LORA="${PERSONA_ADAPTERS:-/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/ddv2_big_final/lora_final}"
NORM="$FMHEAD/traj_norm_stats_ctrl.json"
CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"
CFG_W="${CFG_W:-1.0}"
SELECT="${SELECT:-pdm}"
LIMIT="${LIMIT:-0}"
OUTDIR="${OUTDIR:-/mnt/pfs/zhengguantian/autovla/persona/style_latent_interp_ddv2}"
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$OUTDIR" "$LOG"
[ -f "$HEAD" ]     || { echo "[interp] missing head $HEAD"; exit 2; }
[ -f "$BASE_SRC" ] || { echo "[interp] missing base $BASE_SRC"; exit 2; }

# ---- multi-adapter (task-arithmetic) support: PERSONA_ADAPTERS may be comma-separated ----
# Validate EACH adapter path is a dir (the old `[ -d "$LORA" ]` guard rejected comma-sep strings).
# A single path (no comma) keeps the byte-identical single-LoRA behavior. PERSONA_WEIGHTS defaults
# to EQUAL weights (1/n) -> single adapter is "1.0" (unchanged); a plain N-adapter run is a convex
# blend. When set explicitly, it must have one weight per adapter.
IFS=',' read -r -a _LORA_PATHS <<< "$LORA"
NADP=${#_LORA_PATHS[@]}
for _p in "${_LORA_PATHS[@]}"; do
  [ -d "$_p" ] || { echo "[interp] missing lora dir: $_p"; exit 2; }
done
if [ -n "${PERSONA_WEIGHTS:-}" ]; then
  WEIGHTS="$PERSONA_WEIGHTS"
  IFS=',' read -r -a _W_ARR <<< "$WEIGHTS"
  [ "${#_W_ARR[@]}" -eq "$NADP" ] || { echo "[interp] PERSONA_WEIGHTS (${#_W_ARR[@]}) must match PERSONA_ADAPTERS ($NADP)"; exit 2; }
elif [ "$NADP" -eq 1 ]; then
  WEIGHTS="1.0"                               # single LoRA: byte-identical to before
else
  WEIGHTS="$("$PY" -c "import sys;n=int(sys.argv[1]);print(','.join(['%g'%(1.0/n)]*n))" "$NADP")"
fi

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
  echo "[interp] copying base -> $LOCAL_BASE ..."
  cp "$BASE_SRC" "$LOCAL_BASE" 2>/dev/null || { LOCAL_BASE="/tmp/fm3kin_ep6_base.ckpt"; cp "$BASE_SRC" "$LOCAL_BASE"; }
fi

# ---- PDM metric cache node-local (same fixed pdm+v0 cache as every other run) ----
SRC_CACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
CACHE="${METRIC_CACHE:-/tmp/metric_cache_navtest_v1}"
if [ "$SELECT" = "pdm" ] && [ ! -d "$CACHE/metadata" ]; then
  echo "[interp] copying metric cache -> $CACHE ..."; rm -rf "$CACHE"; cp -r "$SRC_CACHE" "$CACHE"
fi

GPUS=(${GPUS:-0}); N=${#GPUS[@]}
echo "[interp] head=$(basename "$HEAD") base=$BASE_SRC lora=$LORA weights=$WEIGHTS (n=$NADP)"
echo "[interp] alphas=$ALPHAS cfg_w=$CFG_W select=$SELECT limit=$LIMIT gpus=${GPUS[*]} outdir=$OUTDIR"

# ---- (1) sharded KINEMATIC dump, persona LoRA ON, content_source=interp, ALPHAS sweep ----
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" SHARD_IDX="$i" NUM_SHARDS="$N" LIMIT="$LIMIT" \
    BASE_CKPT="$LOCAL_BASE" FMHEAD_CKPT="$HEAD" FMHEAD_NORM="$NORM" \
    PERSONA_ADAPTERS="$LORA" PERSONA_WEIGHTS="$WEIGHTS" FMHEAD_CONTENT_SOURCE=interp \
    FMHEAD_CONSTRAINT="$CONSTRAINT" FMHEAD_STEPS=100 FMHEAD_N=16 FMHEAD_CFG_WEIGHT="$CFG_W" \
    SELECT="$SELECT" METRIC_CACHE="$CACHE" ALPHAS="$ALPHAS" OUTDIR="$OUTDIR" \
    "$PY" "$FMHEAD/dump_fmhead_style.py" > "$LOG/style_interp_shard$i.log" 2>&1 &
  pids+=($!); echo "  shard $i -> GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
FAIL=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  shard $i OK"; else echo "  shard $i FAILED (see $LOG/style_interp_shard$i.log)"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || { echo "[interp] a shard failed -- aborting before merge"; exit 4; }

# ---- (2) merge shards per alpha (no-op when N=1) ----
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
    print(f"[interp] merged alpha={tag}: {len(merged)} tokens -> persona_a{tag}.json")
PYEOF
fi

# ---- (3) full 13-dim style profile + dose-response plot ----
if [ "${SKIP_SWEEP:-0}" != "1" ]; then
  SWEEP_ARGS=(--outdir "$OUTDIR")
  if [ -n "${TEACHER:-}" ]; then
    SWEEP_ARGS+=(--teacher "$TEACHER")
  fi
  "$PY" "$AV/youdrive/sweep_metrics.py" "${SWEEP_ARGS[@]}"
  echo "[interp] DONE -> $OUTDIR/style_metrics.txt"
fi
