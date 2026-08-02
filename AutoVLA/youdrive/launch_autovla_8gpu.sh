#!/bin/bash
# AutoVLA navtest PDMS eval, sharded across 8 GPUs.
# Submit on an 8x A100 node (shares /root/workspace + /mnt/pfs).
set -u
AUTOVLA=/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
cd "$AUTOVLA"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA/navsim"
export PYTHONPATH="$AUTOVLA:$AUTOVLA/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
PY=/root/workspace/miniconda3/envs/autovla/bin/python
NEWCACHE=/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1
CKPT=/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt
CFG=config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml
SF_DIR=navsim/navsim/planning/script/config/common/train_test_split/scene_filter
NSHARD=8
mkdir -p "$AUTOVLA/youdrive/logs"

echo "===== [1/3] generating $NSHARD token shards $(date '+%T') ====="
$PY - "$SF_DIR" "$NSHARD" <<'PYEOF'
import sys, yaml, copy
sf_dir, nshard = sys.argv[1], int(sys.argv[2])
cfg = yaml.safe_load(open(f"{sf_dir}/navtest.yaml"))
tokens = cfg.get("tokens") or []
assert tokens, "navtest.yaml has no tokens list"
for i in range(nshard):
    c = copy.deepcopy(cfg)
    c["tokens"] = tokens[i::nshard]
    yaml.safe_dump(c, open(f"{sf_dir}/navtest_shard{i}.yaml", "w"), default_flow_style=False)
    print(f"  shard{i}: {len(c['tokens'])} tokens")
PYEOF

echo "===== [2/3] launching $NSHARD GPU workers $(date '+%T') ====="
PIDS=()
for i in $(seq 0 $((NSHARD-1))); do
  CUDA_VISIBLE_DEVICES=$i $PY navsim/navsim/planning/script/run_pdm_score_cot.py \
    train_test_split=navtest \
    train_test_split/scene_filter=navtest_shard$i \
    agent=autovla_agent \
    +agent.config_path=$CFG \
    +agent.checkpoint_path=$CKPT \
    +agent.sensor_data_path=. \
    +agent.lora_conf.use_lora=false \
    metric_cache_path=$NEWCACHE \
    json_data_path=dataset/nuplan/navtest_nocot \
    experiment_name=autovla_shard$i \
    > "$AUTOVLA/youdrive/logs/shard$i.log" 2>&1 &
  PIDS+=($!)
  echo "  shard $i -> GPU $i, pid ${PIDS[$i]}"
done

echo "===== waiting for all shards ====="
FAIL=0
for i in $(seq 0 $((NSHARD-1))); do
  if wait "${PIDS[$i]}"; then echo "  shard $i OK"; else echo "  shard $i FAILED"; FAIL=1; fi
done

echo "===== [3/3] merging + scoring $(date '+%T') ====="
$PY - <<'PYEOF'
import glob, csv, os
root = "/mnt/pfs/zhengguantian/autovla/exp"
rows = []
for i in range(8):
    cs = sorted(glob.glob(f"{root}/autovla_shard{i}/*/*.csv"))
    if not cs:
        print(f"  WARN shard{i}: no csv"); continue
    rows += [r for r in csv.DictReader(open(cs[-1]))]
sc = [float(r["score"]) for r in rows if r.get("score") not in (None, "")]
print(f"  merged rows={len(rows)} valid_score={len(sc)}")
if sc:
    print(f"  ===== AutoVLA navtest PDMS (full) = {sum(sc)/len(sc):.4f}  (n={len(sc)}) =====")
    def _m(col):
        v=[float(r[col]) for r in rows if r.get(col) not in (None,"")]
        return sum(v)/len(v) if v else float("nan")
    print("  --- sub-metrics ---")
    for c in ["no_at_fault_collisions","drivable_area_compliance","time_to_collision_within_bound",
              "ego_progress","comfort","driving_direction_compliance","lane_keeping","traffic_light_compliance"]:
        for suf in ["","_stage_one"]:
            k=c+suf
            if any(r.get(k) not in (None,"") for r in rows):
                print(f"    {c:36s} {_m(k):.4f}"); break
PYEOF
echo "ALL_DONE $(date '+%T')"
