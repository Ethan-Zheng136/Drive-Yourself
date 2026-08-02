#!/usr/bin/env bash
# run_dump_fm3kin_viz.sh -- dump fm3-kin ep6 (kinematic decode, pdm-select, v0 from ego
# velocity) trajectories for the first N_TOKENS viz tokens -> RIGHT column of the
# 0.91-vs-fm3-kin 4-panel comparison. GPU (real VLM forward + PDM scoring). Data roots
# from youdrive/eval_persona.sh; python pinned to the autovla env.
set -uo pipefail
AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
KIN_DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42
OUTBASE=/mnt/pfs/zhengguantian/autovla/persona/viz_4panel_091_vs_fm3kin_ep6

cd "$AUTOVLA"
# ---- navsim / nuplan data roots (from youdrive/eval_persona.sh) ----
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# ---- dump config (matches run_fm3_pdms.sh fm3-kin block) ----
export BASE_CKPT="$KIN_DIR/fullft_fm3_kin_ep6_base.ckpt"
export FMHEAD_CKPT="$KIN_DIR/fullft_fm3_kin_ep6_head.pt"
export FMHEAD_NORM="$FMHEAD/traj_norm_stats_ctrl.json"
export TOKENS="$FMHEAD/viz_tokens_240.json"
export N_TOKENS="${N_TOKENS:-200}"
export FMHEAD_N="${FMHEAD_N:-16}"
export FMHEAD_STEPS="${FMHEAD_STEPS:-100}"
export OUT_JSON="${OUT_JSON:-$OUTBASE/right/fm3kin_ep6.json}"

# ---- fast-read: stage base ckpt to node-local (PFS reads are slow for 8GB) ----
SRC_BASE="$BASE_CKPT"
LOCAL_BASE="/dev/shm/fm3kin_ep6_base.ckpt"
if cp "$SRC_BASE" "$LOCAL_BASE" 2>/dev/null; then export BASE_CKPT="$LOCAL_BASE"; else
  LOCAL_BASE="/tmp/fm3kin_ep6_base.ckpt"; cp "$SRC_BASE" "$LOCAL_BASE" && export BASE_CKPT="$LOCAL_BASE"; fi
echo "[dump] base staged -> $BASE_CKPT"

mkdir -p "$(dirname "$OUT_JSON")"
echo "==== [dump fm3-kin ep6] START $(date '+%F %T') | N_TOKENS=$N_TOKENS N=$FMHEAD_N steps=$FMHEAD_STEPS ===="
echo "     GPU: ${CUDA_VISIBLE_DEVICES:-<unset>} | out=$OUT_JSON"
nvidia-smi -L 2>/dev/null | grep -c '^GPU' | sed 's/^/     GPUs visible: /'
"$PY" "$FMHEAD/dump_fm3kin_viz.py"
rc=$?
echo "==== [dump fm3-kin ep6] EXIT rc=$rc $(date '+%F %T') ===="
[ -f "$OUT_JSON" ] && echo "dump: $OUT_JSON ($($PY -c "import json;print(len(json.load(open('$OUT_JSON')))-1)" 2>/dev/null) tokens)"
exit $rc
