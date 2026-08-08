set -x
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/root/workspace/closed_loop/data/navsim/maps"
export OPENSCENE_DATA_ROOT="/root/workspace/closed_loop/data/navsim"
export NAVSIM_EXP_ROOT="/mnt/pfs/zhengguantian/autovla/exp"
export NAVSIM_DEVKIT_ROOT="$(pwd)/navsim"
export PYTHONPATH="$(pwd):$(pwd)/navsim:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
mkdir -p "$NAVSIM_EXP_ROOT"
/root/workspace/miniconda3/envs/autovla/bin/python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/qwen2.5-vl-3B-navtest \
  --output_dir ./dataset/nuplan/navtest_nocot \
  --num_workers 8
echo "PREPROCESS_DONE exit=$?"
