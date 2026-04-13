#!/bin/bash
# SSD experiment environment variables
# Usage: source /home/chokwans99/PSD/ssd/env.sh

export SSD_HF_CACHE="/data2/chokwans99/models"      # HF model root (not hub layout, use direct paths)
export SSD_CUDA_ARCH="8.6"                           # RTX 3090 (H100=9.0, A100=8.0, L40/4090=8.9)

# Direct model paths (bypasses HF hub layout)
export SSD_TARGET_MODEL="/data2/chokwans99/models/layerskip-llama3-8B"
export SSD_DRAFT_MODEL="/data2/chokwans99/models/Llama-3.2-1B-Instruct"

export SSD_DATASET_DIR="/data/ssd_datasets/processed_datasets"
export SSD_DATASET_NUM_SAMPLES="100"

echo "SSD environment loaded."
echo "  Target : $SSD_TARGET_MODEL"
echo "  Draft  : $SSD_DRAFT_MODEL"
echo "  CUDA arch: $SSD_CUDA_ARCH"
