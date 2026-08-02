#!/usr/bin/env bash
# run_fm3_kin_extract_epochs.sh -- extract earlier fm3-kin epochs (ep6, ep10) into
# (base VLM)+(FMHead) using the SAME glue as run_fm3_extract.sh. Distinct per-epoch names
# so /dev/shm base copies never collide. Does NOT touch GOLD/protected weights.
set -euo pipefail
PY=/root/workspace/miniconda3/envs/autovla/bin/python
FMHEAD=/root/workspace/fmhead
RUNDIR=/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42

extract () {  # $1=lightning ckpt  $2=epoch tag (ep6|ep10)
  local ck="$1" ep="$2" dir
  dir="$(dirname "$ck")"
  echo "===== extract fm3-kin $ep @ $(date '+%T') ====="
  echo "  src: $ck"
  $PY "$FMHEAD/extract_full_ft_ckpt.py" --ckpt "$ck" --outdir "$dir"
  mv -f "$dir/full_ft_base.ckpt" "$dir/fullft_fm3_kin_${ep}_base.ckpt"
  mv -f "$dir/fm_decoder.pt"     "$dir/fullft_fm3_kin_${ep}_head.pt"
  echo "  -> base: $dir/fullft_fm3_kin_${ep}_base.ckpt"
  echo "  -> head: $dir/fullft_fm3_kin_${ep}_head.pt"
  ls -la "$dir/fullft_fm3_kin_${ep}_base.ckpt" "$dir/fullft_fm3_kin_${ep}_head.pt"
}

extract "$RUNDIR/epoch=6-loss=0.8686.ckpt"  ep6
extract "$RUNDIR/epoch=10-loss=0.8555.ckpt" ep10
echo "EXTRACT_KIN_EPOCHS_DONE $(date '+%T')"
