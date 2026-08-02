#!/usr/bin/env bash
# run_v2select_overnight.sh -- NEW, additive. Overnight batch of the v2 re-selection headroom
# experiment across base + single teachers + best combos. For each config it (1) dumps all N
# FMHead candidates per token (dump_all mode) via run_pdms_fmhead.sh on 8 GPUs over a strided
# SUBSET_N subset of navtest, then (2) scores every candidate through the v2 EPDMS pipeline
# offline (CPU) and prints the selection CEILING vs per-candidate mean. NO model change, no
# training. Scoring uses the corrected (two_frame-excluded) EPDMS* -> the 0.7 bug cannot recur.
#
# Run:  nohup bash /root/workspace/fmhead/run_v2select_overnight.sh \
#         > /mnt/pfs/zhengguantian/autovla/persona/logs/v2select_overnight.log 2>&1 &
set -uo pipefail
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA

DIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42
V2=/mnt/pfs/zhengguantian/autovla/persona/v2select
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG" "$V2"

# Always keep a PFS copy of the top-level output (platform job log can be empty/torn down).
# MUST be run in the FOREGROUND as the job entry command (NO nohup, NO &) so the batch node
# stays alive for the whole run -- backgrounding makes the job "succeed" in seconds and the
# node (and this script) gets torn down with nothing produced.
exec > >(tee -a "$LOG/v2select_overnight.log") 2>&1
echo "=== run_v2select_overnight START $(date '+%F %T') pid=$$ ==="

# teacher LoRA final dirs
Lddv2=/mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_ddv2/2026-07-20_02-50-59_x8/lora_final
Lgf=/mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_goalflow/2026-07-21_02-13-42_x8/lora_final
Ltf=/mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_transfuser/2026-07-20_02-51-39_x8/lora_final
Lgtrs=/mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_gtrs/2026-07-23_18-13-26_x8/lora_final
Lhydra=/mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_hydra_mdp/2026-07-23_18-13-31_x8/lora_final
Lwote=/mnt/pfs/zhengguantian/autovla/persona/lora_fm3aligned_wote/2026-07-23_19-25-16_x8/lora_final

# common (exported once; the 8GB base is copied to /dev/shm only on the first config)
export BASE_CKPT="$DIR/fullft_fm3_kin_ep6_base.ckpt"
export FMHEAD_CKPT="$DIR/fullft_fm3_kin_ep6_head.pt"
export FMHEAD_NORM=/root/workspace/fmhead/traj_norm_stats_ctrl.json
export FMHEAD_CONSTRAINT='{mode:kinematic,accel_max:5.0,yawrate_max:1.0,v_max:25.0,dt:0.5}'
export FMHEAD_N=16
export SUBSET_N=1200
export DECODER=fmhead
export SELECT=dump_all
export GPUS="0 1 2 3 4 5 6 7"

# name | adapters(comma-sep, empty => base) | weights(comma-sep)
CONFIGS=(
  "base|"
  "ddv2|$Lddv2|1.0"
  "goalflow|$Lgf|1.0"
  "transfuser|$Ltf|1.0"
  "gtrs|$Lgtrs|1.0"
  "hydra|$Lhydra|1.0"
  "wote|$Lwote|1.0"
  "ddv2_gtrs_6040|$Lddv2,$Lgtrs|0.6,0.4"
  "ddv2_hydra_6040|$Lddv2,$Lhydra|0.6,0.4"
  "ddv2_gf_5050|$Lddv2,$Lgf|0.5,0.5"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name adapters weights <<< "$cfg"
  cand="$V2/cand_$name"; out="$V2/reselect_$name"
  echo "############################################################"
  echo "##### [$name] $(date '+%F %T')  adapters='${adapters:-<base>}'"
  echo "############################################################"

  # RESUME: if this config already finished (its reselect log has a CEILING line), skip it.
  # Lets a re-submitted job continue where a torn-down / killed run left off.
  if grep -q "CEILING" "$LOG/reselect_$name.log" 2>/dev/null; then
    echo "[$name] already done (CEILING present) -> skip"
    continue
  fi

  # ---- (1) dump all N candidates (GPU) ----
  if [ -z "$adapters" ]; then
    FMHEAD_STYLE_OFF=true \
      FMHEAD_CAND_DUMP_DIR="$cand" TAG="dumpall_$name" \
      bash /root/workspace/fmhead/run_pdms_fmhead.sh > "$LOG/dump_$name.log" 2>&1 \
      || { echo "[$name] DUMP FAILED (see $LOG/dump_$name.log)"; continue; }
  else
    FMHEAD_STYLE_OFF=false FMHEAD_CONTENT_SOURCE=interp FMHEAD_ALPHA=1.0 \
      PERSONA_ADAPTERS="$adapters" PERSONA_WEIGHTS="$weights" \
      FMHEAD_CAND_DUMP_DIR="$cand" TAG="dumpall_$name" \
      bash /root/workspace/fmhead/run_pdms_fmhead.sh > "$LOG/dump_$name.log" 2>&1 \
      || { echo "[$name] DUMP FAILED (see $LOG/dump_$name.log)"; continue; }
  fi
  echo "[$name] dumped $(ls "$cand" 2>/dev/null | wc -l) tokens"

  # ---- (2) offline v2 re-selection (CPU) ----
  bash /root/workspace/fmhead/v2_reselect_experiment.sh "$cand" "$out" 16 > "$LOG/reselect_$name.log" 2>&1 \
    || { echo "[$name] RESELECT FAILED (see $LOG/reselect_$name.log)"; continue; }

  echo "----- [$name] RESULT -----"
  grep -E "tokens \(all-candidate|mean over candidates|WORST|CEILING" "$LOG/reselect_$name.log"
done

echo "############################################################"
echo "##### ALL v2select CONFIGS DONE $(date '+%F %T')"
echo "##### summary (CEILING per config):"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name _ _ <<< "$cfg"
  line=$(grep "CEILING" "$LOG/reselect_$name.log" 2>/dev/null | head -1)
  echo "  $name : ${line:-<no result>}"
done
echo "############################################################"
