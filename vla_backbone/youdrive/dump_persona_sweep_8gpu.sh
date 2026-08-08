#!/usr/bin/env bash
# Full YDSP dose-response dump: base + alpha*v_DDv2 over ALL 4049 styletest tokens, 8 GPUs.
# alpha=0 is NOT recomputed (== base AutoVLA == compare_full/autovla.json).
# Usage: ADAPTER=/path/lora_final bash youdrive/dump_persona_sweep_8gpu.sh
set -uo pipefail
AV=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AV"
export PYTHONPATH="$AV:$AV/navsim:${PYTHONPATH:-}"
ADAPTER="${ADAPTER:?set ADAPTER=lora adapter dir}"
ALPHAS="${ALPHAS:-0.5,1.0}"
OUTDIR="${OUTDIR:-/mnt/pfs/zhengguantian/autovla/compare_persona}"
PY=/root/workspace/miniconda3/envs/autovla/bin/python
LOG=/mnt/pfs/zhengguantian/autovla/persona/logs; mkdir -p "$LOG" "$OUTDIR"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}

# avoid pfs I/O storm: copy the 16GB base ckpt to node-local once, all shards read local
SRC_BASE=/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt
LOCAL_BASE="${LOCAL_BASE:-/dev/shm/autovla_base.ckpt}"
if [ ! -f "$LOCAL_BASE" ]; then
  echo "[sweep8] copying base ckpt -> $LOCAL_BASE ..."
  if ! cp "$SRC_BASE" "$LOCAL_BASE" 2>/dev/null; then
    LOCAL_BASE=/tmp/autovla_base.ckpt; echo "[sweep8] /dev/shm failed, using $LOCAL_BASE"; cp "$SRC_BASE" "$LOCAL_BASE"
  fi
fi
export BASE_CKPT="$LOCAL_BASE"

echo "[sweep8] adapter=$ADAPTER alphas=$ALPHAS gpus=${GPUS[*]} base=$BASE_CKPT"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} SHARD_IDX=$i NUM_SHARDS=$N \
  ADAPTER="$ADAPTER" ALPHAS="$ALPHAS" OUTDIR="$OUTDIR" WEIGHTS="${WEIGHTS:-1}" \
  "$PY" youdrive/dump_persona_sweep.py > "$LOG/sweep8_shard$i.log" 2>&1 &
  pids+=($!); echo "  shard $i -> GPU ${GPUS[$i]} (pid ${pids[-1]})"
done
for p in "${pids[@]}"; do wait "$p"; done

# merge shards per alpha
"$PY" - "$OUTDIR" "$ALPHAS" "$N" <<'PYEOF'
import json, glob, sys, os
outdir, alphas, n = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
for a in alphas:
    a=float(a); merged={}; meta=None
    for f in sorted(glob.glob(os.path.join(outdir, f"persona_a{a}.shard*.json"))):
        d=json.load(open(f)); meta=meta or d.get("_meta")
        for k,v in d.items():
            if k!="_meta": merged[k]=v
    out={"_meta":meta} if meta else {}; out.update(merged)
    json.dump(out, open(os.path.join(outdir, f"persona_a{a}.json"),"w"))
    print(f"  merged alpha={a}: {len(merged)} tokens -> persona_a{a}.json")
PYEOF
echo "[sweep8] done -> $OUTDIR/persona_a*.json"
