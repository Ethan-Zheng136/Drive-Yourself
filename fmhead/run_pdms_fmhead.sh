#!/usr/bin/env bash
# run_pdms_fmhead.sh -- navtest EPDMS/PDMS for the feasibility comparison, reusing the
# EXISTING navsim PDMS harness (run_pdm_score_cot.py + youdrive/_agg_pdms.py).
#
# Runs the SAME base AutoVLA weights with EITHER decoder, so the two rows of the
# feasibility table are directly comparable:
#   DECODER=codebook  -> stock AutoVLAAgent (AutoVLA.predict, argmax codebook)  [baseline]
#   DECODER=fmhead    -> FMHeadAutoVLAAgent (FMHead sample + select, s=0)       [ours]
#
# Usage:
#   DECODER=codebook TAG=cb_base bash run_pdms_fmhead.sh
#   DECODER=fmhead   TAG=fm_gt   FMHEAD_CKPT=/mnt/pfs/.../fmhead_ckpts_gt/<run>/fmhead_final.pt \
#     bash run_pdms_fmhead.sh
#   SMOKE=1 ... (20 tokens, 1 GPU)
#
# Style axis OFF (feasibility): no PERSONA_ADAPTERS. Requires an NVML-enabled node
# (real vision forward) + navtest metric cache (as in youdrive/eval_persona.sh).
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
DECODER="${DECODER:-fmhead}"
TAG="${TAG:-$DECODER}"
CFG=config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml
SF_DIR=navsim/navsim/planning/script/config/common/train_test_split/scene_filter
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG"

# BASE_CKPT overrides the base VLM the harness loads (default = AutoVLA_PDMS_89, the
# frozen-eval base). For full-FT eval pass the extracted trained base:
#   BASE_CKPT=/mnt/pfs/.../fmhead_sft_full/<run>/full_ft_base.ckpt
SRC_BASE="${BASE_CKPT:-/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt}"
SRC_CACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
# node-local copy path differs per base so full-FT and frozen runs don't clobber each other
CKPT="${LOCAL_BASE:-/dev/shm/autovla_base_$(basename "$SRC_BASE" .ckpt).ckpt}"
[ -f "$CKPT" ] || cp "$SRC_BASE" "$CKPT" 2>/dev/null || { CKPT=/tmp/$(basename "$CKPT"); cp "$SRC_BASE" "$CKPT"; }
echo "[pdms] base VLM = $SRC_BASE"
CACHE="${LOCAL_CACHE:-/tmp/metric_cache_navtest_v1}"
[ -d "$CACHE/metadata" ] || { rm -rf "$CACHE"; cp -r "$SRC_CACHE" "$CACHE"; }

# decoder-specific hydra overrides
EXTRA_AGENT=()
if [ "$DECODER" = "fmhead" ]; then
  : "${FMHEAD_CKPT:?set FMHEAD_CKPT=/path/to/fmhead_final.pt}"
  NORM="${FMHEAD_NORM:-$FMHEAD/traj_norm_stats_gt.json}"
  # SELECT: medoid (default) | pdm (score N candidates with navsim PDM, argmax) |
  #         oracle (eval-only: min-ADE-to-GT candidate = upper bound)
  SELECT="${SELECT:-medoid}"
  # STYLE controllability: set FMHEAD_STYLE_OFF=false + PERSONA_ADAPTERS=<ddv2 lora> +
  # FMHEAD_ALPHA=<a> (+ FMHEAD_CFG=<w>) to eval PDMS with the persona style axis ON at gain
  # alpha. Default (feasibility) keeps style OFF (s=0). PERSONA_ADAPTERS is read by the stock
  # AutoVLAAgent.initialize (env-based LoRA attach); the agent then decouples content(base)/style(delta).
  STYLE_OFF="${FMHEAD_STYLE_OFF:-true}"
  EXTRA_AGENT=(
    "agent._target_=fmhead_navsim_agent.FMHeadAutoVLAAgent"
    "+agent.fmhead_ckpt_path=$FMHEAD_CKPT"
    "+agent.fmhead_normalizer_path=$NORM"
    "+agent.fmhead_style_off=$STYLE_OFF"
    "+agent.fmhead_style_alpha=${FMHEAD_ALPHA:-0.0}"
    "+agent.fmhead_num_samples=${FMHEAD_N:-16}"
    "+agent.fmhead_num_steps=${FMHEAD_STEPS:-30}"
    "+agent.fmhead_cfg_weight=${FMHEAD_CFG:-1.0}"
    "+agent.fmhead_select_mode=$SELECT"
    "+agent.fmhead_metric_cache_path=$CACHE"
    "+agent.fmhead_use_style_adapter=${FMHEAD_USE_STYLE_ADAPTER:-false}"
  )
  # dump_all diagnostic: when FMHEAD_CAND_DUMP_DIR is set (with SELECT=dump_all), the agent
  # writes ALL N candidates per token there for OFFLINE re-selection. Additive: unset -> no-op.
  if [ -n "${FMHEAD_CAND_DUMP_DIR:-}" ]; then
    EXTRA_AGENT+=("+agent.fmhead_candidate_dump_dir=$FMHEAD_CAND_DUMP_DIR")
    echo "[pdms] candidate dump dir = $FMHEAD_CAND_DUMP_DIR (SELECT=$SELECT)"
  fi
  # fm3: propagate the decode-time feasibility constraint (kinematic unicycle / soft).
  # WITHOUT this, a control-space (fm3-kin) head decodes its (a_long,yaw_rate) samples as
  # raw xy -> garbage. Pass a Hydra dict, e.g.
  #   FMHEAD_CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
  # Empty (default) -> byte-identical fm2 xy decode (mode none), so fm3-lite needs nothing.
  if [ -n "${FMHEAD_CONSTRAINT:-}" ]; then
    EXTRA_AGENT+=("+agent.fmhead_constraint=$FMHEAD_CONSTRAINT")
    echo "[pdms] fm3 constraint = $FMHEAD_CONSTRAINT"
  fi
  # interp latent-content mode: c = h_base + alpha*(h_styled-h_base), s=0 (needs PERSONA_ADAPTERS
  # for the LoRA-on forward). Empty (default) -> unchanged (base/styled legacy behavior).
  if [ -n "${FMHEAD_CONTENT_SOURCE:-}" ]; then
    EXTRA_AGENT+=("+agent.fmhead_content_source=$FMHEAD_CONTENT_SOURCE")
    echo "[pdms] content_source = $FMHEAD_CONTENT_SOURCE"
  fi
  # learned scorer (SELECT=learned): load the trained deployable scorer to pick 1 of N candidates
  # from ctx+poses ONLY (no metric cache, no GT future). Additive: unset -> no-op.
  if [ -n "${FMHEAD_SCORER_CKPT:-}" ]; then
    EXTRA_AGENT+=("+agent.fmhead_scorer_ckpt_path=$FMHEAD_SCORER_CKPT")
    EXTRA_AGENT+=("+agent.fmhead_scorer_weights=${FMHEAD_SCORER_WEIGHTS:-v1}")
    echo "[pdms] learned scorer ckpt = $FMHEAD_SCORER_CKPT (weights=${FMHEAD_SCORER_WEIGHTS:-v1})"
  fi
fi

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
if [ "${SMOKE:-0}" = "1" ]; then GPUS=(0); NSHARD=1; NTOK=20; else NSHARD=${#GPUS[@]}; NTOK=0; fi
echo "[pdms] DECODER=$DECODER TAG=$TAG shards=$NSHARD gpus=${GPUS[*]}"

# BUG-2 fix (a): fail LOUDLY up front if the requested GPUs aren't actually visible.
# (A prior run had shards 4-7 hit "No CUDA GPUs are available" and the aggregator then
#  printed a misleading partial number on ~half the scenes.)
# Count only real "GPU N:" lines (nvidia-smi -L also prints indented "  MIG ... Device"
# sub-lines on MIG nodes; the old `wc -l` over-counted those).
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
echo "[pdms] nvidia-smi -L shows $NGPU GPU(s); requesting indices: ${GPUS[*]}"
if [ "$NGPU" -lt 1 ]; then
  echo "[pdms][FATAL] no GPUs visible (nvidia-smi -L empty). Aborting."; exit 3
fi
for g in "${GPUS[@]}"; do
  if ! [[ "$g" =~ ^[0-9]+$ ]]; then
    echo "[pdms][FATAL] GPU spec '$g' is not an integer index (GPUS='${GPUS[*]}'). Aborting."; exit 3
  fi
  if [ "$g" -ge "$NGPU" ]; then
    echo "[pdms][FATAL] requested GPU index $g but only $NGPU GPU(s) visible (0..$((NGPU-1))). Aborting."; exit 3
  fi
done

# Per-device diagnostics + a real CUDA-visibility probe. This is the EVIDENCE that
# distinguishes "the device is genuinely absent/busy for this shard's process" (env
# problem, would also break the stock agent) from "our agent init broke device access"
# (a code bug). torch.cuda.device_count()==0 under a valid CUDA_VISIBLE_DEVICES=$gpu is a
# genuine environment/allocation problem, NOT something our agent controls.
gpu_diag() {  # $1=gpu index -> one-line nvidia-smi memory summary for that device
  local gpu="$1"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$gpu" 2>&1 \
    | head -1 | sed 's/^/    nvidia-smi: /'
}
probe_cuda() {  # $1=gpu index -> echoes probe line, returns 0 if CUDA sees >=1 device
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="$gpu" $PY -c \
    "import torch,sys; n=torch.cuda.device_count(); print('    cuda_probe: CUDA_VISIBLE_DEVICES=%s -> device_count=%d'%('$gpu',n)); sys.exit(0 if n>0 else 7)" 2>&1
}

$PY - "$SF_DIR" "$NSHARD" "$NTOK" "$TAG" <<'PYEOF'
import sys, os, yaml, copy
sf_dir, nshard, ntok, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = yaml.safe_load(open(f"{sf_dir}/navtest.yaml")); toks = cfg.get("tokens") or []
if ntok > 0: toks = toks[:ntok]
# SUBSET_N: evaluate a STRIDED representative subset of navtest (single-GPU tractable).
# Strided (toks[::stride]) spans the whole set -> a fair PDMS estimate, unlike first-N.
sub = int(os.environ.get("SUBSET_N", "0"))
if sub > 0 and sub < len(toks):
    stride = max(1, len(toks) // sub)
    toks = toks[::stride][:sub]
    print(f"  SUBSET_N={sub} -> strided {len(toks)} tokens (stride={stride})")
for i in range(nshard):
    c = copy.deepcopy(cfg); c["tokens"] = toks[i::nshard]
    yaml.safe_dump(c, open(f"{sf_dir}/fmfeas_{tag}_shard{i}.yaml", "w"))
    print(f"  shard{i}: {len(c['tokens'])} tokens")
PYEOF

run_shard() {  # $1=shard idx, $2=gpu ; returns run_pdm exit code, log -> $LOG
  # BUG-A fix: guard positionals with ${x-} (empty, not unbound) so `set -u` can NEVER
  # abort here with "$2: unbound variable"; validate instead. An EMPTY gpu would set
  # CUDA_VISIBLE_DEVICES="" -> torch would report "No CUDA GPUs are available", so we
  # refuse to launch with a blank device (that was a silent BUG-B trigger too).
  local i="${1-}" gpu="${2-}" lg="$LOG/pdms_${TAG}_shard${1-unknown}.log"
  if [ -z "$i" ] || [ -z "$gpu" ]; then
    echo "[pdms][FATAL] run_shard called with i='$i' gpu='$gpu' (both required)"; return 3
  fi
  # per-shard diagnostics header (captured in the shard log for post-hoc evidence)
  { echo "==== shard $i @ $(date '+%T') ===="
    echo "    CUDA_VISIBLE_DEVICES(requested)=$gpu"
    gpu_diag "$gpu"
    probe_cuda "$gpu"; echo "    cuda_probe_rc=$?"
  } >> "$lg" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" $PY navsim/navsim/planning/script/run_pdm_score_cot.py \
    train_test_split=navtest \
    train_test_split/scene_filter=fmfeas_${TAG}_shard$i \
    agent=autovla_agent +agent.config_path=$CFG +agent.checkpoint_path=$CKPT \
    +agent.sensor_data_path=. +agent.lora_conf.use_lora=false \
    "${EXTRA_AGENT[@]}" \
    metric_cache_path=$CACHE json_data_path=dataset/nuplan/navtest_nocot \
    experiment_name=fmfeas_${TAG}_shard$i >> "$lg" 2>&1
}

pids=()
for i in $(seq 0 $((NSHARD-1))); do
  gpu="${GPUS[$i]}"
  # Fail-fast, per-shard CUDA visibility EVIDENCE before spending minutes loading a VLM.
  echo "  shard $i -> GPU $gpu :"
  gpu_diag "$gpu"
  if probe_cuda "$gpu"; then :; else
    echo "  [pdms][DIAG] shard $i GPU $gpu: CUDA sees 0 devices in a fresh process."
    echo "  [pdms][DIAG] -> device genuinely unavailable to THIS process (env/allocation),"
    echo "  [pdms][DIAG]    not an agent bug (stock agent would fail here too)."
  fi
  run_shard "$i" "$gpu" & pids+=($!); echo "  shard $i launched pid ${pids[-1]}"
done

# BUG-2 fix (b/optional-c): collect per-shard exit codes; retry-once with backoff on the
# transient "No CUDA GPUs available" flake; a non-zero shard marks the WHOLE run failed.
# OOM is reported separately (it is NOT the same failure class and is not retried blindly).
FAIL=0; FAILED_SHARDS=""
for i in $(seq 0 $((NSHARD-1))); do
  if wait "${pids[$i]}"; then
    echo "  shard $i OK"
  else
    lg="$LOG/pdms_${TAG}_shard$i.log"
    if grep -qiE "out of memory|CUDA out of memory" "$lg"; then
      echo "  shard $i FAILED (CUDA OUT OF MEMORY -- not a device-visibility issue; reduce FMHEAD_N or free the GPU)"
      FAIL=1; FAILED_SHARDS+=" $i"
    elif grep -q "No CUDA GPUs are available" "$lg"; then
      echo "  shard $i hit 'No CUDA GPUs' -> re-probing GPU ${GPUS[$i]} then retrying once (5s backoff)"
      sleep 5
      if probe_cuda "${GPUS[$i]}"; then
        if run_shard "$i" "${GPUS[$i]}"; then echo "  shard $i OK (retry)"; else
          echo "  shard $i FAILED (after retry)"; FAIL=1; FAILED_SHARDS+=" $i"; fi
      else
        echo "  shard $i: GPU ${GPUS[$i]} STILL has 0 CUDA devices on re-probe -> GENUINE node/allocation problem (not our agent)."
        FAIL=1; FAILED_SHARDS+=" $i"
      fi
    else
      echo "  shard $i FAILED (exit!=0; see $lg)"; FAIL=1; FAILED_SHARDS+=" $i"; fi
  fi
done

# Verify each shard actually produced a CSV (belt-and-suspenders vs silent partials)
NCSV=0
for i in $(seq 0 $((NSHARD-1))); do
  if ls "$NAVSIM_EXP_ROOT"/fmfeas_${TAG}_shard$i/*/*.csv >/dev/null 2>&1; then NCSV=$((NCSV+1)); fi
done
echo "[pdms] shards with CSV: $NCSV/$NSHARD ; failed shards:${FAILED_SHARDS:- none}"

# STRICT aggregation: --require-all makes agg refuse to print a trusted PDMS unless ALL
# NSHARD shards produced CSVs; otherwise it prints 'PARTIAL k/N - DO NOT TRUST' and exits !=0.
$PY "$FMHEAD/agg_pdms_fmhead.py" "$TAG" "$NSHARD" "$NAVSIM_EXP_ROOT" fmfeas --require-all
AGG_RC=$?
if [ "$FAIL" -ne 0 ] || [ "$NCSV" -ne "$NSHARD" ] || [ "$AGG_RC" -ne 0 ]; then
  echo "[pdms][RESULT] FAILED / PARTIAL (fail=$FAIL csv=$NCSV/$NSHARD agg_rc=$AGG_RC) -- DO NOT TRUST any number above."
  echo "ALL_DONE $(date '+%T')"; exit 4
fi
echo "[pdms][RESULT] OK - all $NSHARD shards produced CSVs."
echo "ALL_DONE $(date '+%T')"
