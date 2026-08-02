#!/usr/bin/env bash
# run_eval_feasibility.sh -- robust launcher for eval_feasibility.py (decoded-traj vs GT L2).
#
# WHY: eval_feasibility.py crashed with `ModuleNotFoundError: No module named
# 'pytorch_lightning'` because it was launched with the WRONG interpreter
# (/opt/conda python3.11) instead of the project env. The code/imports are fine; the fix
# is the interpreter + PYTHONPATH. This wrapper pins the autovla env python and paths
# (mirrors run_pdms_fmhead.sh's header), so `bash run_eval_feasibility.sh ...` always works.
#
# Usage (real vision node, after training):
#   FMHEAD_CKPT=/mnt/pfs/.../fmhead_ckpts_gt/<run>/fmhead_final.pt \
#     bash run_eval_feasibility.sh --n 500
#   # extra args are forwarded to eval_feasibility.py, e.g. --no_codebook, --smoke
# Text-only smoke on a restricted box:
#   FMHEAD_TEXT_ONLY=1 FMHEAD_CKPT=<ckpt> bash run_eval_feasibility.sh --n 4 --smoke --no_sft_base --no_codebook
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# HARD-PIN the project interpreter (this is the actual fix for the ModuleNotFoundError).
PY=/root/workspace/miniconda3/envs/autovla/bin/python
CONFIG="${CONFIG:-config/fmhead_gt_feasibility.yaml}"
: "${FMHEAD_CKPT:?set FMHEAD_CKPT=/path/to/fmhead_final.pt}"

# BASE_CKPT: override the base VLM (default keeps the config's AutoVLA_PDMS_89). For
# full-FT eval, pass the extracted trained base: BASE_CKPT=/mnt/pfs/.../full_ft_base.ckpt
BASE_ARG=()
if [ -n "${BASE_CKPT:-}" ]; then BASE_ARG=(--base_ckpt "$BASE_CKPT"); fi

echo "[eval] python=$PY"
echo "[eval] config=$CONFIG ckpt=$FMHEAD_CKPT base_ckpt=${BASE_CKPT:-<config default PDMS_89>} extra_args=$*"
"$PY" "$FMHEAD/eval_feasibility.py" --config "$FMHEAD/$CONFIG" --ckpt "$FMHEAD_CKPT" "${BASE_ARG[@]}" "$@"
