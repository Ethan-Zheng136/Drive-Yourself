#!/bin/bash
# NuScenes Dataset Preprocessing Script for DriveVLA
# Generates training and validation data from NuScenes dataset.

set -e

# Default paths - modify these for your setup
NUSCENES_PATH="./dataset/nuscenes" # /path/to/nuscenes
OUTPUT_BASE_DIR="./dataset/nuscenes" # /path/to/output 
NUSCENES_VERSION="v1.0-trainval"
DRIVELM_PATH=""  # Optional: path to DriveLM annotations /path/to/drivelm/v1_1_train_nus.json

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nuscenes_path)
            NUSCENES_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --version)
            NUSCENES_VERSION="$2"
            shift 2
            ;;
        --drivelm_path)
            DRIVELM_PATH="$2"
            shift 2
            ;;
        --train-only)
            PROCESS_VAL=false
            shift
            ;;
        --val-only)
            PROCESS_TRAIN=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --nuscenes_path PATH    Path to NuScenes dataset (default: /data/dataset/nuscenes)"
            echo "  --output_dir PATH       Output directory (default: data/nuscenes)"
            echo "  --version VERSION       NuScenes version (default: v1.0-trainval)"
            echo "  --drivelm_path PATH     Optional path to DriveLM annotations"
            echo "  --train-only            Only process training split"
            echo "  --val-only              Only process validation split"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: process both train and val
PROCESS_TRAIN=${PROCESS_TRAIN:-true}
PROCESS_VAL=${PROCESS_VAL:-true}

echo "================================================"
echo "NuScenes Dataset Preprocessing"
echo "================================================"
echo "NuScenes Path: $NUSCENES_PATH"
echo "Output Directory: $OUTPUT_BASE_DIR"
echo "Version: $NUSCENES_VERSION"
echo "DriveLM Path: ${DRIVELM_PATH:-'(not provided)'}"
echo "Process Train: $PROCESS_TRAIN"
echo "Process Val: $PROCESS_VAL"
echo "================================================"

# Build optional DriveLM argument (only for training data)
DRIVELM_ARG=""
if [ -n "$DRIVELM_PATH" ]; then
    DRIVELM_ARG="--drivelm_path $DRIVELM_PATH"
fi

# Process training data (with DriveLM if provided)
if [ "$PROCESS_TRAIN" = true ]; then
    echo ""
    echo "Processing NuScenes training data..."
    python tools/preprocessing/nusc_sample_generation.py \
        --nuscenes_path "$NUSCENES_PATH" \
        --output_dir "${OUTPUT_BASE_DIR}_train" \
        --split train \
        --version "$NUSCENES_VERSION" \
        $DRIVELM_ARG
fi

# Process validation data (no DriveLM - it's training data only)
if [ "$PROCESS_VAL" = true ]; then
    echo ""
    echo "Processing NuScenes validation data..."
    python tools/preprocessing/nusc_sample_generation.py \
        --nuscenes_path "$NUSCENES_PATH" \
        --output_dir "${OUTPUT_BASE_DIR}_val" \
        --split val \
        --version "$NUSCENES_VERSION"
fi

echo ""
echo "================================================"
echo "NuScenes preprocessing complete!"
if [ "$PROCESS_TRAIN" = true ]; then
    echo "Training data saved to: ${OUTPUT_BASE_DIR}_train"
fi
if [ "$PROCESS_VAL" = true ]; then
    echo "Validation data saved to: ${OUTPUT_BASE_DIR}_val"
fi
echo "================================================"
