#!/bin/bash
export PYTHONPATH=./navsim:$PYTHONPATH

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="/path/to/your/checkpoint.ckpt"  # AutoVLA_PDMS_89.ckpt        # downloaded from HuggingFace
CACHE_PATH="./dataset/nuplan/navtest_metric_cache"
JSON_DATA_PATH="./dataset/nuplan/navtest_nocot"
SENSOR_DATA_PATH="./dataset/nuplan/sensor_blobs/test"
CONFIG_PATH="$./config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
LORA=false

CUDA_VISIBLE_DEVICES=0 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  agent=autovla_agent \
  +agent.config_path="$CONFIG_PATH" \
  +agent.checkpoint_path="$CHECKPOINT" \
  +agent.sensor_data_path="$SENSOR_DATA_PATH" \
  +agent.lora_conf.use_lora=$LORA \
  metric_cache_path=$CACHE_PATH \
  json_data_path=$JSON_DATA_PATH \
  experiment_name=autovla_agent\