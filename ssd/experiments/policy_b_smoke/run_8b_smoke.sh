#!/bin/bash
# Smoke test: layerskip-llama3-8B (fp16, TP=4) + TinyLlama draft.
# Goal: verify Policy B unified path runs to completion without crash + sane output.
# Minimal numseqs/output_len to fit on busy shared GPUs.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_PROFILE_MESA=1
export SSD_FORCE_SPLIT_K1K2=1
export SSD_TRACE_SPLIT_K1K2=1   # enable verbose policy=b traces for verification

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"
echo "[host-load] $(uptime | awk -F'load average:' '{print $2}')"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT="$PWD/experiments/policy_b_smoke/out_8b"
mkdir -p "$OUT"

# Minimal: 2 seqs, 16 output tokens, temp=0 (deterministic)
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0 --output_len 16 --max_model_len 1024 --random --numseqs 2"

echo "[$(date +%H:%M:%S)] === 8B smoke (Policy B default) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
SSD_PROFILE_DIR="$OUT" \
    "$PY" -O bench/bench.py $COMMON \
    --k 16 --f 3 --mesa --mesa_phase1_k 8 --mesa_phase2_k 8 \
    --mesa_draft_fan_out 2 \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
err=$(grep -oP 'Traceback|RuntimeError|FAIL|Aborted|AssertionError' "$OUT/run.log" | head -3)
echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"

echo ""
echo "===== correctness signals ====="
grep -E "MESA-SSD enabled|mesa_proxy_top_k raised|mesa_policy|Total Throughput|Avg Tokens per step|Avg Fraction|Avg Phase|Avg Cache" "$OUT/run.log" | head -15

echo ""
echo "===== first few traces (TRACE-split-k1k2) ====="
grep "TRACE-split-k1k2" "$OUT/run.log" | head -10

echo ""
echo "===== first error (if any) ====="
grep -E "Error|Traceback" "$OUT/run.log" | head -10
