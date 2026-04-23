#!/bin/bash
# Phase 5: persistent artifact smoke test
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12299
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT=/home/chokwans99/PSD/ssd/tmp/int4_phase5
ART=$OUT/art_llama3_8b_int4

pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 14000 {print $1}' | head -$1 | paste -sd,; }
GPUS="2,3,4"
[ -z "$GPUS" ] && { echo "no GPUs"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPUS

COMMON="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --temp 0.6 --output_len 48 --max_model_len 512 --all --numseqs 2 --draft_path $DRAFT --async --spec --k 7 --flh 5 4 4 3 3 2 2 1 --quant_int4"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""
    echo "[$(date +%H:%M:%S)] === $tag ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    local t0=$SECONDS
    "$PY" -O bench/bench.py "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local elapsed=$((SECONDS - t0))
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local acc=$(grep "Avg Fraction" "$subdir/run.log" | grep -oP '[\d.]+' | tail -1)
    echo "  rc=$rc  wall=${elapsed}s  tp=${tp:-FAIL}  acc=${acc:-n/a}"
}

# Clean any stale artifact
rm -f $ART.*.pt

# First run: quantize + save artifact
run "1_first_dump" $COMMON --quant_artifact "$ART"

# Verify files created
echo ""
ls -lh $ART.*.pt 2>&1 | head -5

# Second run: load from artifact (should be faster startup)
run "2_second_load" $COMMON --quant_artifact "$ART"

# Third run: --quant_artifact_load_only (strict)
run "3_load_only" $COMMON --quant_artifact "$ART" --quant_artifact_load_only

echo ""
echo "=== SUMMARY ==="
for t in 1_first_dump 2_second_load 3_load_only; do
    log="$OUT/$t/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    saved=$(grep -c "saved rank" "$log" 2>/dev/null)
    loaded=$(grep -c "loaded rank" "$log" 2>/dev/null)
    printf "%-25s TP=%-8s acc=%s saved=%s loaded=%s\n" "$t" "${tp:-FAIL}" "${acc:-n/a}" "$saved" "$loaded"
done
