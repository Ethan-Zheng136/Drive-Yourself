#!/usr/bin/env bash
# run_fm3_jerk.sh -- jerk/feasibility audit for BOTH fm3 heads (kin first: it exercises the
# never-full-run kinematic decode path, so we catch any error in minutes, not after PDMS).
set -uo pipefail
AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FMHEAD=/root/workspace/fmhead
cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:$FMHEAD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
PY=/root/workspace/miniconda3/envs/autovla/bin/python
N="${N:-600}"
OUT=/mnt/pfs/zhengguantian/autovla/persona/fm3_eval

LITE_DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_lite/2026-07-17_13-15-12
KIN_DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42

echo "########## fm3-kin jerk audit (kinematic decode) $(date '+%T') ##########"
CUDA_VISIBLE_DEVICES=0 $PY "$FMHEAD/eval_jerk_feasibility.py" --variant fm3_kin \
  --base_ckpt "$KIN_DIR/fullft_fm3_kin_base.ckpt" \
  --fmhead_ckpt "$KIN_DIR/fullft_fm3_kin_head.pt" \
  --normalizer "$FMHEAD/traj_norm_stats_ctrl.json" \
  --n "$N" --num_samples 16 --num_steps 100 --out "$OUT"
KIN_RC=$?
echo "fm3-kin jerk rc=$KIN_RC"

echo "########## fm3-lite jerk audit (xy decode) $(date '+%T') ##########"
CUDA_VISIBLE_DEVICES=0 $PY "$FMHEAD/eval_jerk_feasibility.py" --variant fm3_lite \
  --base_ckpt "$LITE_DIR/fullft_fm3_lite_base.ckpt" \
  --fmhead_ckpt "$LITE_DIR/fullft_fm3_lite_head.pt" \
  --normalizer "$FMHEAD/traj_norm_stats_gt.json" \
  --n "$N" --num_samples 16 --num_steps 100 --out "$OUT"
LITE_RC=$?
echo "fm3-lite jerk rc=$LITE_RC"
echo "JERK_ALL_DONE kin_rc=$KIN_RC lite_rc=$LITE_RC $(date '+%T')"
