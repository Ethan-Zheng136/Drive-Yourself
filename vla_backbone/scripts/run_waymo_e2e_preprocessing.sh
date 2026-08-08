#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

INCLUDE_COT=false
CONFIG="dataset/qwen2.5-vl-72B-waymo"
OUTPUT_DIR="./dataset/waymo"

if [ "$INCLUDE_COT" = true ]; then
    echo "Preprocessing with Chain-of-Thought (CoT)..."
    CUDA_VISIBLE_DEVICES=0,1 python tools/preprocessing/cot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Preprocessing without Chain-of-Thought (No-CoT)..."
    python tools/preprocessing/nocot_sample_generation.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
fi