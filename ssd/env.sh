#!/bin/bash
# SSD experiment environment variables
# Usage: source /home/chokwans99/PSD/ssd/env.sh
#
# Target box: 8x RTX 4090 24GB (sm_89), CUDA 12.8, /home/chokwans99.
# GPU 0-1 are frequently held by another user -- check nvidia-smi before
# picking CUDA_VISIBLE_DEVICES. The 70B AWQ layout below assumes 2-7.

export SSD_HF_CACHE="/data/chokwans99/models"        # flat dir, not HF hub layout: use direct paths below
export SSD_CUDA_ARCH="8.9"                           # RTX 4090 (H100=9.0, A100=8.0, RTX PRO 6000=12.0)
export SSD_ATTN_BACKEND="auto"                       # sm_89 -> sgl-kernel (flashinfer is the sm_120 path only)

# Direct model paths (bypasses HF hub layout)
export SSD_TARGET_MODEL="/data/chokwans99/models/layerskip-llama2-70B"
export SSD_DRAFT_MODEL="/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0"

# paths.py expects <ds>_data_10000.jsonl; this dir holds correctly-named
# symlinks to the 100-row files in /data/ssd_datasets. 100 prompts per
# dataset -- keep --numseqs <= 100. See that dir's README.txt.
export SSD_DATASET_DIR="/home/chokwans99/ssd_datasets/processed_datasets"

# AWQ W4A16 artifacts. /data has <25GB free, so these live on / (~270GB free).
export SSD_AWQ_CALIB_DIR="/home/chokwans99/awq_calibrated"
export SSD_AWQ_ARTIFACT_DIR="/home/chokwans99/awq_artifacts"

echo "SSD environment loaded."
echo "  Target : $SSD_TARGET_MODEL"
echo "  Draft  : $SSD_DRAFT_MODEL"
echo "  CUDA arch: $SSD_CUDA_ARCH"
