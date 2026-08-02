#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# Default values
DATASET_FOLDER="./dataset/waymo_open_dataset_end_to_end_camera_v_1_0_1"
OUTPUT_FOLDER="${DATASET_FOLDER}/test_images"
SPLIT="test"

echo "Running Waymo E2E format transfer..."
echo "Dataset folder: $DATASET_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Split: $SPLIT"

python tools/preprocessing/waymo_e2e_image_extraction.py \
    --dataset_folder "$DATASET_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --split "$SPLIT"