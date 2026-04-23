#!/bin/bash
# 8B MESA long run comparison: dense vs INT4 (no_quant_lm_head)
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT=/home/chokwans99/PSD/ssd/tmp/int4_mesa_long

# Use 3 GPUs — matches earlier MESA 8B setup (TP=2 + draft)
pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 14000 {print $1}' | head -$1 | paste -sd,; }
GPUS=$(pick 3)
[ -z "$GPUS" ] && { echo "no GPUs"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPUS
echo "GPUs=$GPUS"

COMMON="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 15 --draft_path $DRAFT --async --spec --k 5 --f 4 --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""
    echo "[$(date +%H:%M:%S)] === $tag ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local acc=$(grep "Avg Fraction" "$subdir/run.log" | grep -oP '[\d.]+' | tail -1)
    echo "  rc=$rc tp=${tp:-FAIL} acc=${acc:-n/a}"
}

run "1_mesa_8b_dense_long"
run "2_mesa_8b_int4_long" --quant_int4 --no_quant_lm_head

echo ""
echo "=== SUMMARY ==="
for t in 1_mesa_8b_dense_long 2_mesa_8b_int4_long; do
    log="$OUT/$t/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    time_s=$(grep -oP 'Time:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    printf "%-30s TP=%-8s accept=%-6s time=%ss\n" "$t" "${tp:-FAIL}" "${acc:-n/a}" "${time_s:-?}"
done
