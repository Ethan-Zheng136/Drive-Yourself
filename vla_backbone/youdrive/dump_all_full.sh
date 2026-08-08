#!/usr/bin/env bash
# Dump ALL 4 models' trajectories over the 4049 styletest tokens, 8 GPUs each, sequential.
# Each model -> /mnt/pfs/zhengguantian/autovla/compare_full/<model>.json
# Usage:
#   bash dump_all_full.sh              # run all four
#   bash dump_all_full.sh transfuser   # run a single model
#   SMOKE=1 bash dump_all_full.sh      # 2 tokens/shard sanity run
set -uo pipefail

# Code lives in the workspace; only large data (token lists, trajectory dumps) live under /mnt/pfs.
WS=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive
ROOT=/mnt/pfs/zhengguantian/autovla   # data root (outputs + token lists)
OUTDIR=$ROOT/compare_full
SHARDS=$OUTDIR/shards
LOGDIR=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/youdrive/dump_logs
mkdir -p "$OUTDIR" "$LOGDIR"
# GPU list: space-separated device ids. Default 8 GPUs; override e.g. GPUS="0" on single-GPU nodes.
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
N=${#GPUS[@]}
TOKENS_FULL=/mnt/pfs/zhengguantian/autovla/compare/tokens_styletest.json

# common navsim data env
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/root/workspace/closed_loop/data/navsim/maps
export OPENSCENE_DATA_ROOT=/root/workspace/closed_loop/data/navsim
export HF_HUB_OFFLINE=1

DD_CKPT=/mnt/pfs/zhengguantian/output/g_csa_vad/ckpts/diffusiondrive/diffusiondrive_navsim_88p1_PDMS
DDV2_CKPT=/root/workspace/closed_loop/DiffusionDriveV2/ckpts/diffusiondrivev2_sel.ckpt

mkdir -p "$SHARDS"
# split the full token list into N per-model shards (N = number of GPUs).
# Per-model filenames so concurrently-submitted model jobs never collide.
# SMOKE=1 caps each shard to 2 tokens for a sanity run.
split_tokens() {  # $1 = model
  python3 -c "
import json
toks=json.load(open('$TOKENS_FULL')); N=$N; smoke=${SMOKE:-0}; m='$1'
for i in range(N):
    s=toks[i::N]
    if smoke: s=s[:2]
    json.dump(s, open('$SHARDS/%s_shard%d.json'%(m,i),'w'))
print('  [%s] shards: N=%d sizes=%s'%(m,N,[len(toks[i::N]) if not smoke else min(2,len(toks[i::N])) for i in range(N)]))"
}

merge() {  # $1 = model name
  python3 - "$1" <<'PYEOF'
import json, glob, os, sys
name=sys.argv[1]; outdir="/mnt/pfs/zhengguantian/autovla/compare_full"
merged={"_meta":{"model":name,"frame":"ego BEV, x=forward(m), y=left(m), cumulative positions"}}
nf=0
for f in sorted(glob.glob(os.path.join(outdir,f"{name}_shard*.json"))):
    d=json.load(open(f)); m=d.get("_meta",{})
    merged["_meta"]["horizon_s"]=m.get("horizon_s"); merged["_meta"]["n_pts"]=m.get("n_pts")
    nf+=len(m.get("failed",[]))
    for k,v in d.items():
        if k!="_meta": merged[k]=v
json.dump(merged, open(os.path.join(outdir,f"{name}.json"),"w"))
print(f"[merge] {name}: tokens={len(merged)-1} failed={nf}")
PYEOF
}

# generic 8-GPU sharded launch.  args: model py script cwd  (env-var wiring inside)
launch() {
  local model=$1 py=$2 script=$3 cwd=$4
  split_tokens "$model"
  echo "=== $model: launching $N shards on GPUs ${GPUS[*]} ==="
  local pids=()
  for i in $(seq 0 $((N-1))); do
    local gpu=${GPUS[$i]}
    local TOK=$SHARDS/${model}_shard${i}.json
    local OUT=$OUTDIR/${model}_shard${i}.json
    ( cd "$cwd" && \
      PYTHONPATH="$cwd:$cwd/navsim:${PYTHONPATH:-}" \
      CUDA_VISIBLE_DEVICES=$gpu \
      TOKENS_FILE="$TOK" TOKENS_PATH="$TOK" TOKENS_JSON="$TOK" \
      OUT_FILE="$OUT" OUT_PATH="$OUT" OUT_JSON="$OUT" \
      CKPT="${CKPT:-}" AGENT_CFG="${AGENT_CFG:-}" \
      NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT/exp}" \
      "$py" "$script" ) > "$LOGDIR/${model}_shard${i}.log" 2>&1 &
    pids+=($!); echo "  shard $i -> GPU $gpu (pid ${pids[-1]})  log: $LOGDIR/${model}_shard${i}.log"
  done
  for p in "${pids[@]}"; do wait "$p"; done
  merge "$model"
}

run_autovla() {
  # env aligned with the proven launch_autovla_8gpu.sh
  local A=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
  CKPT= AGENT_CFG= \
  NAVSIM_DEVKIT_ROOT="$A/navsim" \
  NAVSIM_EXP_ROOT=/mnt/pfs/zhengguantian/autovla/exp \
  TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 \
  launch autovla /root/workspace/miniconda3/envs/autovla/bin/python \
    "$WS/dump_autovla.py" \
    "$A"
}
run_diffusiondrive() {
  CKPT=$DD_CKPT AGENT_CFG= \
  launch diffusiondrive /root/workspace/miniconda3/envs/diffusiondrive/bin/python \
    /root/workspace/closed_loop/diffusiondrive/dump_traj.py \
    /root/workspace/closed_loop/diffusiondrive
}
run_diffusiondrivev2() {
  CKPT=$DDV2_CKPT AGENT_CFG=diffusiondrivev2_sel_agent \
  launch diffusiondrivev2 /root/workspace/miniconda3/envs/ddv2/bin/python \
    /root/workspace/closed_loop/DiffusionDriveV2/dump_traj.py \
    /root/workspace/closed_loop/DiffusionDriveV2
}
run_transfuser() {
  CKPT= AGENT_CFG= NAVSIM_EXP_ROOT=/mnt/pfs/zhengguantian/transfuser/exp \
  launch transfuser /root/workspace/miniconda3/envs/navsim/bin/python \
    "$WS/run_transfuser_tokens.py" \
    /root/workspace/closed_loop/navsim
}

TARGET=${1:-all}
case "$TARGET" in
  autovla) run_autovla ;;
  diffusiondrive) run_diffusiondrive ;;
  diffusiondrivev2) run_diffusiondrivev2 ;;
  transfuser) run_transfuser ;;
  all) run_autovla; run_diffusiondrive; run_diffusiondrivev2; run_transfuser ;;
  *) echo "unknown target: $TARGET"; exit 1 ;;
esac
echo "ALL DONE -> $OUTDIR/{autovla,diffusiondrive,diffusiondrivev2,transfuser}.json"
