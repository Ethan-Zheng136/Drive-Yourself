#!/usr/bin/env bash
# run_fm3_extract.sh -- split the two fm3 full-FT Lightning ckpts into (base VLM)+(FMHead),
# following the GOLD pattern (fullft_fm2_trainedVLM_base.ckpt + fullft_fm2_head_pdm0.9088.pt).
# Artifacts land next to each run dir with distinct, descriptive names (so /dev/shm base
# copies never collide between variants). Does NOT touch GOLD/protected weights.
set -euo pipefail
PY=/root/workspace/miniconda3/envs/autovla/bin/python
FMHEAD=/root/workspace/fmhead

LITE_CKPT="/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_lite/2026-07-17_13-15-12/epoch=14-loss=0.2334.ckpt"
KIN_CKPT="/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42/epoch=14-loss=0.8346.ckpt"

extract () {  # $1=lightning ckpt  $2=tag (lite|kin)
  local ck="$1" tag="$2" dir
  dir="$(dirname "$ck")"
  echo "===== extract fm3-$tag @ $(date '+%T') ====="
  echo "  src: $ck"
  $PY "$FMHEAD/extract_full_ft_ckpt.py" --ckpt "$ck" --outdir "$dir"
  mv -f "$dir/full_ft_base.ckpt" "$dir/fullft_fm3_${tag}_base.ckpt"
  mv -f "$dir/fm_decoder.pt"     "$dir/fullft_fm3_${tag}_head.pt"
  echo "  -> base: $dir/fullft_fm3_${tag}_base.ckpt"
  echo "  -> head: $dir/fullft_fm3_${tag}_head.pt"
  ls -la "$dir/fullft_fm3_${tag}_base.ckpt" "$dir/fullft_fm3_${tag}_head.pt"
}

extract "$LITE_CKPT" lite
extract "$KIN_CKPT"  kin
echo "EXTRACT_ALL_DONE $(date '+%T')"
