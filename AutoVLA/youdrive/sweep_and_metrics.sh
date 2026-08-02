#!/usr/bin/env bash
# One command: dump persona sweep trajectories (8 GPU) THEN auto-compute & SAVE the
# YDSP + kinematic + social style metrics (no manual hand-back of raw JSON needed).
#
# Stage 1: youdrive/dump_persona_sweep_8gpu.sh  -> OUTDIR/persona_a{ALPHA}.json
# Stage 2: youdrive/sweep_metrics.py            -> OUTDIR/style_metrics.{json,txt,png}
#                                                  (+ OUTDIR/env_interact.json)
#
# Usage:
#   ADAPTER=/path/lora_final OUTDIR=/mnt/pfs/zhengguantian/autovla/persona/my_sweep \
#     bash youdrive/sweep_and_metrics.sh
# Optional:
#   ALPHAS=0.5,0.75,1.0   GPUS="0 1 2 3 4 5 6 7"   SKIP_DUMP=1 (metrics only on existing dumps)
#
# The original youdrive/dump_persona_sweep_8gpu.sh remains usable standalone.
set -uo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AV"
export PYTHONPATH="$AV:$AV/navsim:${PYTHONPATH:-}"
# NUPLAN env for stage-2 env_interact (social metrics); values mirror train_grpo_persona.sh L17-22
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/root/workspace/closed_loop/data/navsim/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/root/workspace/closed_loop/data/navsim}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$AV/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/pfs/zhengguantian/autovla/exp}"

PY=/root/workspace/miniconda3/envs/autovla/bin/python
ALPHAS="${ALPHAS:-0.5,1.0}"
OUTDIR="${OUTDIR:-/mnt/pfs/zhengguantian/autovla/compare_persona}"
SKIP_DUMP="${SKIP_DUMP:-0}"
mkdir -p "$OUTDIR"

if [ "$SKIP_DUMP" != "1" ]; then
  echo "[sweep_and_metrics] STAGE 1: GPU dump -> $OUTDIR"
  ADAPTER="${ADAPTER:?set ADAPTER=lora adapter dir (or SKIP_DUMP=1 for metrics-only)}" \
  ALPHAS="$ALPHAS" OUTDIR="$OUTDIR" GPUS="${GPUS:-0 1 2 3 4 5 6 7}" \
    bash youdrive/dump_persona_sweep_8gpu.sh
else
  echo "[sweep_and_metrics] SKIP_DUMP=1 -> using existing dumps in $OUTDIR"
fi

echo "[sweep_and_metrics] STAGE 2: style metrics -> $OUTDIR/style_metrics.{json,txt,png}"
"$PY" youdrive/sweep_metrics.py --outdir "$OUTDIR" --alphas "$ALPHAS"
echo "[sweep_and_metrics] done. metrics saved under $OUTDIR"
