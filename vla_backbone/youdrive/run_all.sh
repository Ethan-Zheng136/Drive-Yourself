set -e
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$(pwd)/navsim"
export PYTHONPATH="$(pwd):$(pwd)/navsim:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
PY=/root/workspace/miniconda3/envs/autovla/bin/python
mkdir -p dataset/nuplan/navtest_nocot "$NAVSIM_EXP_ROOT"

echo "===== STEP1 PREPROCESS $(date '+%T') ====="
$PY tools/preprocessing/nocot_sample_generation.py \
  --config dataset/qwen2.5-vl-3B-navtest --output_dir ./dataset/nuplan/navtest_nocot --num_workers 8
N=$(ls dataset/nuplan/navtest_nocot/*.json 2>/dev/null | wc -l)
echo "PREPROCESS_DONE json=$N"
[ "$N" -lt 100 ] && { echo "TOO_FEW_JSON abort"; exit 1; }

echo "===== STEP2 EVAL $(date '+%T') ====="
CUDA_VISIBLE_DEVICES=0 $PY navsim/navsim/planning/script/run_pdm_score_cot.py \
  train_test_split=navtest \
  agent=autovla_agent \
  +agent.config_path=config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml \
  +agent.checkpoint_path=/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt \
  +agent.sensor_data_path=dataset/nuplan/sensor_blobs/test \
  +agent.lora_conf.use_lora=false \
  metric_cache_path=dataset/nuplan/navtest_metric_cache \
  json_data_path=dataset/nuplan/navtest_nocot \
  experiment_name=autovla_agent
echo "EVAL_DONE $(date '+%T')"
