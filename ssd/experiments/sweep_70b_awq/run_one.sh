#!/bin/bash
# Single-run wrapper for the 70B AWQ + TinyLlama AWQ sweep.
# Usage: bash run_one.sh <out_subdir> <extra_bench_args...>
#
# Always uses 5 GPUs (4 target TP + 1 draft) on CUDA_VISIBLE_DEVICES=0..4.
# Caller is expected to have already verified GPUs 0..4 are free.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
# Caller may override SWEEP_GPUS (comma-separated 5-GPU list); else auto-pick free GPUs.
if [ -z "${SWEEP_GPUS:-}" ]; then
    SWEEP_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F', ' '{gsub(" MiB","",$2); if ($2+0 < 500) print $1}' \
        | head -5 | paste -sd ,)
fi
export CUDA_VISIBLE_DEVICES="$SWEEP_GPUS"
echo "[run_one] using CUDA_VISIBLE_DEVICES=$SWEEP_GPUS"
export SSD_PROFILE_MESA=1

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

# Per-run port pinned in caller via $SSD_DIST_PORT (avoid collisions).
: "${SSD_DIST_PORT:?must set SSD_DIST_PORT}"

OUT_SUBDIR="$1"; shift
SUBDIR="$SCRIPT_DIR/$OUT_SUBDIR"
mkdir -p "$SUBDIR"

fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
sleep 2

echo "[$(date +%H:%M:%S)] === ${OUT_SUBDIR} === $* (port=${SSD_DIST_PORT})"

SSD_PROFILE_DIR="$SUBDIR" "$PY" -O bench/bench.py \
  --llama --size 8 --model_path "$TARGET" --draft_path "$DRAFT" \
  --quant_awq --quant_awq_artifact "$TGT_ART" \
  --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
  --gpus 5 --b 1 --temp 0.6 --max_model_len 2048 --all \
  "$@" >"$SUBDIR/run.log" 2>&1

EXIT=$?
TP=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$SUBDIR/run.log" | head -1)
echo "[$(date +%H:%M:%S)] ← ${OUT_SUBDIR}: exit=$EXIT TP=${TP:-FAIL}"
exit $EXIT
