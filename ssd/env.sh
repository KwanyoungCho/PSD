#!/bin/bash
# SSD experiment environment variables
# Usage: source /home/chokwans99/PSD/ssd/env.sh

export SSD_HF_CACHE="/data"               # required by paths.py (모델 캐시 루트)
export SSD_CUDA_ARCH="8.9"                # RTX 4090 (H100=9.0, A100=8.0)

export SSD_DATASET_DIR="/data/ssd_datasets/processed_datasets"
export SSD_DATASET_NUM_SAMPLES="100"      # 100개 샘플 (나중에 10000으로 변경)

echo "SSD environment loaded."
echo "  HF cache : $SSD_HF_CACHE"
echo "  CUDA arch: $SSD_CUDA_ARCH"
echo "  Datasets : $SSD_DATASET_DIR ($SSD_DATASET_NUM_SAMPLES samples)"
