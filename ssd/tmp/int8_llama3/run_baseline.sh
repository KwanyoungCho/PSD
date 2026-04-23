#!/bin/bash
# Dense baselines for async spec + MESA, to compute delta vs int8.
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
OUT=/home/chokwans99/PSD/ssd/tmp/int8_llama3

pick_gpus() {
    local n=$1
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '$2 > 14000 {print $1}' | head -$n | paste -sd,
}

GPUS=$(pick_gpus 3)
[ -z "$GPUS" ] && { echo "no GPUs"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPUS
echo "using GPUs: $GPUS"

COMMON="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --output_len 48 --max_model_len 512 --all --numseqs 2 --draft_path $DRAFT --spec --k 7 --temp 0.6 --async"

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
    if [ $rc -ne 0 ] || [ -z "$tp" ]; then
        echo "  -> FAIL (rc=$rc)"
        return 1
    fi
    echo "  -> OK  TP=$tp  accept=${acc:-n/a}"
}

# Dense async spec sampling baseline
run "6b_async_dense_sampling" --flh 5 4 4 3 3 2 2 1

# Dense MESA baseline
run "7b_mesa_dense_sampling" --f 4 --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2 --k 5

echo ""
echo "=== BASELINE SUMMARY ==="
for d in 6b_async_dense_sampling 7b_mesa_dense_sampling; do
    log="$OUT/$d/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "  %-30s TP=%-8s accept=%s\n" "$d" "${tp:-FAIL}" "${acc:-n/a}"
done
