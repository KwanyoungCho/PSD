#!/bin/bash
# 8B model comparison: AR vs Sync SD vs Async SSD (1B) vs Async SSD (EAGLE3)
# Option C: Target TP=4 fixed, async gets +1 for draft

set -e
cd /home/chokwans99/PSD/ssd/bench

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export SSD_HF_CACHE=/data
export SSD_DATASET_DIR=/data/ssd_datasets/processed_datasets
export SSD_DATASET_NUM_SAMPLES=100
export SSD_CUDA_ARCH=8.9

PYTHON=/home/chokwans99/anaconda3/envs/ssd/bin/python
COMMON="--llama --size 8 --b 1 --temp 0 --numseqs 100 --output_len 256 --all"
RESULTS=/home/chokwans99/PSD/ssd/results
mkdir -p $RESULTS

echo "============================================"
echo "[1/4] AR Baseline (TP=4, 4 GPU)"
echo "============================================"
$PYTHON -O bench.py $COMMON --gpus 4 \
  2>&1 | tee $RESULTS/cmp_8b_ar.log

echo "============================================"
echo "[2/4] Sync SD - 1B draft (TP=4, 4 GPU)"
echo "============================================"
$PYTHON -O bench.py $COMMON --gpus 4 --spec --k 7 \
  2>&1 | tee $RESULTS/cmp_8b_sync_1b.log

echo "============================================"
echo "[3/4] Async SSD - 1B draft (TP=4 + 1 draft, 5 GPU)"
echo "============================================"
$PYTHON -O bench.py $COMMON --gpus 5 --spec --async --k 7 --f 3 \
  2>&1 | tee $RESULTS/cmp_8b_async_1b.log

echo "============================================"
echo "[4/4] Async SSD - EAGLE3 draft (TP=4 + 1 draft, 5 GPU)"
echo "============================================"
$PYTHON -O bench.py $COMMON --gpus 5 --eagle --async --k 7 --f 3 \
  2>&1 | tee $RESULTS/cmp_8b_async_eagle3.log

echo ""
echo "============================================"
echo "ALL DONE. Results in $RESULTS/"
echo "============================================"
