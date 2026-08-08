#!/usr/bin/env bash
# Combine the navtrain + navtest DDv2 teacher SFT JSONs into ONE dir via SYMLINKS (CPU, idempotent).
#
# NON-DESTRUCTIVE: creates P/nocot_ddv2_teacher_final and symlinks every *.json from BOTH source
# dirs into it. No source file is copied, modified, or deleted. Re-runnable: existing correct
# symlinks are left alone; only missing ones are (re)created. Run once now to link the navtrain
# half (~11988), then AGAIN after the navtest teacher build (step 3) to add the navtest half.
#
# Note on sensor roots: navtrain teacher JSONs embed ABSOLUTE camera paths (sensor_data_path is
# ignored via os.path.join), while navtest teacher JSONs (copied from navtest_nocot) embed paths
# RELATIVE to the AutoVLA repo root. The _final SFT/GRPO configs therefore set sensor_data_path to
# the repo root so BOTH resolve. See the config comments.
set -uo pipefail
P=/mnt/pfs/zhengguantian/autovla/persona
TRAIN_DIR=${TRAIN_DIR:-$P/nocot_ddv2_teacher12k}
TEST_DIR=${TEST_DIR:-$P/nocot_ddv2_teacher_navtest}
OUT_DIR=${OUT_DIR:-$P/nocot_ddv2_teacher_final}
case "$OUT_DIR" in
  /mnt/pfs/*) : ;;
  *) echo "ERROR: OUT_DIR must be on PFS, got $OUT_DIR" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"

link_dir() {  # $1 = source dir (may not exist yet)
  local src=$1
  if [ ! -d "$src" ]; then
    echo "[combine] NOTE: source dir not present yet, skipping: $src"
    return 0
  fi
  local n=0
  shopt -s nullglob
  for f in "$src"/*.json; do
    local dst="$OUT_DIR/$(basename "$f")"
    if [ -L "$dst" ]; then
      # already a symlink; leave it (idempotent). Fail loud only on a real-file collision.
      continue
    fi
    if [ -e "$dst" ]; then
      echo "ERROR: $dst exists and is not a symlink; refusing to clobber" >&2
      exit 1
    fi
    ln -s "$f" "$dst"
    n=$((n+1))
  done
  shopt -u nullglob
  echo "[combine] linked $n new json(s) from $src"
}

link_dir "$TRAIN_DIR"
link_dir "$TEST_DIR"
echo "[combine] total json symlinks in $OUT_DIR: $(find "$OUT_DIR" -maxdepth 1 -name '*.json' | wc -l)"
