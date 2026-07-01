#!/usr/bin/env bash
# Single async-SD run for the 7B + AMD-135m sweep.
# Args: K F GPUS PORT
#   GPUS: CUDA_VISIBLE_DEVICES value (e.g. "0,1")
#   PORT: SSD_DIST_PORT (must be unique across concurrent runs)
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 K F GPUS PORT" >&2
  exit 2
fi
K=$1
F=$2
GPUS=$3
PORT=$4

ROOT=/home/chokwans99/PSD/ssd
SWEEP_DIR=$ROOT/experiments/proxy_async_overlap/async_sd_sweep_7b_amd135m
PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
OUTDIR=$SWEEP_DIR/k${K}_f${F}${OUT_TAG:-}
mkdir -p "$OUTDIR"

echo "[$(date -Is)] === START k=${K} f=${F} gpus=${GPUS} port=${PORT} ===" | tee "$OUTDIR/run.log"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" \
SSD_DIST_PORT="$PORT" \
SSD_PROFILE_MESA="${SSD_PROFILE_MESA:-0}" \
SSD_PROFILE_DIR="$OUTDIR" \
SSD_CUDA_ARCH=8.6 \
TORCH_CUDA_ARCH_LIST=8.6 \
SSD_HF_CACHE=/data2/chokwans99/models \
SSD_DATASET_DIR=/data2/chokwans99/datasets \
MPLCONFIGDIR=/tmp/matplotlib \
"$PY" -O bench/bench.py \
  --llama --size 8 \
  --model_path /data2/chokwans99/models/layerskip-llama2-7B \
  --draft_path /data2/chokwans99/models/AMD-Llama-135m-fp16 \
  --gpus 2 --b 1 --temp 0.7 --seed 42 --numseqs 50 \
  --input_len 512 --output_len 512 --all --max_model_len 2048 \
  --async --spec --k "$K" --f "$F" \
  >> "$OUTDIR/run.log" 2>&1
status=$?

if [[ $status -ne 0 ]]; then
  echo "[$(date -Is)] === FAILED k=${K} f=${F} status=${status} ===" | tee -a "$OUTDIR/run.log"
else
  echo "[$(date -Is)] === END k=${K} f=${F} ===" | tee -a "$OUTDIR/run.log"
fi

{
  echo "k=${K} f=${F} status=${status}"
  grep -E "Total Throughput|Final Decode Throughput|Avg target time|Avg Tokens per step|Avg Fraction of Speculated|Avg Cache Hits" "$OUTDIR/run.log" || true
} > "$OUTDIR/headline.txt"

exit 0
