#!/usr/bin/env bash
# run_diagnostics.sh -- evaluate an FMHead ckpt on navtest.
# DEFAULT = deployable pdm-selection PDMS + L2 (no oracle, since oracle peeks at GT
# and is slow — only useful as a one-off ceiling diagnostic).
#
# Usage:
#   # frozen head (default base = AutoVLA_PDMS_89):
#   FMHEAD_CKPT=/path/to/fmhead_final.pt bash run_diagnostics.sh
#   # full-FT head (pass the extracted trained VLM base):
#   FMHEAD_CKPT=/path/fm_decoder.pt BASE_CKPT=/path/full_ft_base.ckpt bash run_diagnostics.sh
#   # also run the oracle ceiling (slow, uses GT):  RUN_ORACLE=1 ... bash run_diagnostics.sh
#   # tag / gpus overridable:  TAG=myrun GPUS="0 1 2 3" ...
set -uo pipefail
FM=/root/workspace/fmhead
: "${FMHEAD_CKPT:?set FMHEAD_CKPT=/path/to/fmhead(_final|_decoder).pt}"
TAG="${TAG:-fmeval}"
GPUS="${GPUS:-0 1 2 3}"
STEPS="${FMHEAD_STEPS:-30}"
N="${FMHEAD_N:-16}"
# optional trained-VLM base override (full-FT); empty => script default (AutoVLA_PDMS_89)
BASE_ARG=""; [ -n "${BASE_CKPT:-}" ] && BASE_ARG="BASE_CKPT=$BASE_CKPT"

echo "########## PDM-selection PDMS (deployable, no GT) ##########"
env DECODER=fmhead TAG="${TAG}_pdm" SELECT=pdm GPUS="$GPUS" FMHEAD_CKPT="$FMHEAD_CKPT" \
    FMHEAD_STEPS="$STEPS" FMHEAD_N="$N" $BASE_ARG bash "$FM/run_pdms_fmhead.sh"

echo "########## L2 vs human GT (fast) ##########"
env FMHEAD_CKPT="$FMHEAD_CKPT" $BASE_ARG bash "$FM/run_eval_feasibility.sh" --n 500 --no_codebook

if [ "${RUN_ORACLE:-0}" = "1" ]; then
  echo "########## ORACLE selection (ceiling, uses GT — slow, opt-in) ##########"
  env DECODER=fmhead TAG="${TAG}_oracle" SELECT=oracle GPUS="$GPUS" FMHEAD_CKPT="$FMHEAD_CKPT" \
      FMHEAD_STEPS="$STEPS" FMHEAD_N="$N" $BASE_ARG bash "$FM/run_pdms_fmhead.sh"
fi

echo "########## DONE (${TAG}) — read the PDM-selection number above ##########"
