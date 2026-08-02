#!/bin/bash
export PYTHONPATH=./navsim:$PYTHONPATH

TRAIN_TEST_SPLIT=warmup_test_e2e
CACHE_PATH=./dataset/nuplan/warmup_test_e2e_cache

# TRAIN_TEST_SPLIT=navtest
# CACHE_PATH=./dataset/nuplan/navtest_metric_cache
# NAVSIM_EXP_ROOT="./"

export OPENSCENE_DATA_ROOT="./dataset/nuplan"  # Path to the OpenScene dataset
export NUPLAN_MAPS_ROOT="./dataset/nuplan/maps"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH