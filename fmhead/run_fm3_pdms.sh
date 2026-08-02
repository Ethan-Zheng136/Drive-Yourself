#!/usr/bin/env bash
# run_fm3_pdms.sh -- navtest PDMS (pdm-select, N=16) for BOTH fm3 heads on ONE MIG GPU.
# pdm-select is REQUIRED: medoid mode-averaging collapses PDMS (~0.25); GOLD 0.9088 was
# pdm-select. Single MIG slice -> strided SUBSET_N-scene navtest subset (tractable, fair).
# Reuses run_pdms_fmhead.sh (hardened shard/aggregate logic + strict --require-all).
set -uo pipefail
FMHEAD=/root/workspace/fmhead
SUBSET_N="${SUBSET_N:-3000}"
export SUBSET_N GPUS="0"

LITE_DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_lite/2026-07-17_13-15-12
KIN_DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42

echo "########## fm3-lite PDMS (pdm-select, subset=$SUBSET_N) $(date '+%T') ##########"
DECODER=fmhead TAG=fm3_lite SELECT=pdm FMHEAD_N=16 FMHEAD_STEPS=100 \
  BASE_CKPT="$LITE_DIR/fullft_fm3_lite_base.ckpt" \
  FMHEAD_CKPT="$LITE_DIR/fullft_fm3_lite_head.pt" \
  FMHEAD_NORM="$FMHEAD/traj_norm_stats_gt.json" \
  bash "$FMHEAD/run_pdms_fmhead.sh"
echo "fm3-lite PDMS rc=$?"

echo "########## fm3-kin PDMS (pdm-select, kinematic decode, subset=$SUBSET_N) $(date '+%T') ##########"
DECODER=fmhead TAG=fm3_kin SELECT=pdm FMHEAD_N=16 FMHEAD_STEPS=100 \
  BASE_CKPT="$KIN_DIR/fullft_fm3_kin_base.ckpt" \
  FMHEAD_CKPT="$KIN_DIR/fullft_fm3_kin_head.pt" \
  FMHEAD_NORM="$FMHEAD/traj_norm_stats_ctrl.json" \
  FMHEAD_CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}' \
  bash "$FMHEAD/run_pdms_fmhead.sh"
echo "fm3-kin PDMS rc=$?"
echo "PDMS_ALL_DONE $(date '+%T')"
