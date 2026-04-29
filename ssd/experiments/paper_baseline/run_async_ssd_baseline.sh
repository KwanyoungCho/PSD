#!/bin/bash
# Paper baseline: async SSD with uniform k=7, f=3 on layerskip-llama2-70B (AWQ TP=4)
# + TinyLlama-1B (AWQ) draft. Uses --all (union of 4 datasets, numseqs*4 total).
#
# Per memory: temp=0.7 is the paper standard. seed=42 for reproducibility.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"
echo "[host-load] $(uptime | awk -F'load average:' '{print $2}')"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

# numseqs (override-able): 10 = trend check; final paper run uses 128.
NUMSEQS="${NUMSEQS:-10}"
NAME="${NAME:-ssd_baseline_k7_f3_temp07_all_len256_seed42_n${NUMSEQS}}"
OUT="$PWD/experiments/paper_baseline/${NAME}"
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] === ${NAME} (k=7, f=3, temp=0.7, seed=42, n=${NUMSEQS}) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2

"$PY" -O bench/bench.py \
    --llama --size 70 \
    --gpus 5 \
    --spec --async \
    --k 7 --f 3 \
    --b 1 \
    --temp 0.7 \
    --seed 42 \
    --numseqs "$NUMSEQS" \
    --output_len 256 \
    --all \
    --name "${NAME}" \
    --model_path "$TARGET" \
    --draft_path "$DRAFT" \
    --quant_awq --quant_awq_artifact "$TGT_ART" --quant_group_size 128 \
    --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
err=$(grep -oP 'Traceback|RuntimeError|FAIL|Aborted|AssertionError|OutOfMemoryError' "$OUT/run.log" | head -2)
echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"

echo ""
echo "===== metrics ====="
grep -E "Avg Tokens per step|Avg Fraction|Avg target verify|Avg draft step|Total Throughput|Final Prefill|Final Decode|Total: " "$OUT/run.log" | head -10
