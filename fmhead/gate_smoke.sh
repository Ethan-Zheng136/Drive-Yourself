#!/usr/bin/env bash
# gate_smoke.sh -- cheap cluster smoke gate BEFORE committing to the full retrain.
# Gate A: frozen with-vision training smoke (1 GPU, 50 steps) on the REAL node
#         (load_sft_base from config => base == AutoVLA_PDMS_89, consistent w/ PDMS harness).
# Gate B: SMOKE PDMS (1 GPU, 20 tokens) on Gate-A's ckpt -> proves the agent runs
#         end-to-end, produces NON-nan scores, aligned-c no longer collapses (plumbing).
# It is a PLUMBING gate, not a performance gate (50 steps is undertrained).
#
# Usage:  bash /root/workspace/fmhead/gate_smoke.sh
set -uo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
FM=/root/workspace/fmhead
PY=/root/workspace/miniconda3/envs/autovla/bin/python
cd "$AV"
export PYTHONPATH="$AV:$AV/navsim:$FM:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
CKDIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts_gt

echo "=================== GATE A: frozen with-vision smoke (1 GPU, 50 steps) ==================="
CUDA_VISIBLE_DEVICES=0 "$PY" "$FM/train_fmhead.py" \
  --config config/fmhead_gt_feasibility.yaml --smoke --max_steps 50 \
  || { echo "!!! GATE A FAILED (training smoke) !!!"; exit 1; }

CK=$(ls -dt "$CKDIR"/*_smoke/fmhead_final.pt 2>/dev/null | head -1)
[ -f "$CK" ] || { echo "!!! GATE A: no ckpt produced under $CKDIR/*_smoke !!!"; exit 1; }
echo "=== Gate A OK. ckpt = $CK ==="

echo "=================== GATE B: SMOKE PDMS (1 GPU, 20 tokens) ==================="
SMOKE=1 DECODER=fmhead TAG=fmgate FMHEAD_CKPT="$CK" bash "$FM/run_pdms_fmhead.sh" \
  || { echo "!!! GATE B FAILED (PDMS agent) !!!"; exit 1; }

echo "=================== GATE A+B PASSED — safe to launch full retrain ==================="
echo "Next: CONFIG=config/fmhead_gt_feasibility.yaml bash $FM/launch_8gpu.sh   (frozen, ~1.7h)"
echo "      CONFIG=config/fmhead-sft-full.yaml       bash $FM/launch_sft_full_8gpu.sh (full-FT)"
