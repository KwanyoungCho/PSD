#!/bin/bash
# Phase 6: CodeLlama-34B int4 scaling
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=$(ls -d /data2/chokwans99/models/models--facebook--layerskip-codellama-34B/snapshots/*/ | head -1)
TARGET="${TARGET%/}"
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
OUT=/home/chokwans99/PSD/ssd/tmp/int4_phase6

pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 18000 {print $1}' | head -$1 | paste -sd,; }

GPUS=$(pick 5)
[ -z "$GPUS" ] || [ $(echo $GPUS | tr ',' '\n' | wc -l) -lt 5 ] && { echo "need 5 GPUs, got: $GPUS"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPUS
echo "GPUs: $GPUS"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""
    echo "[$(date +%H:%M:%S)] === $tag ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    "$PY" -O bench/bench.py "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local acc=$(grep "Avg Fraction" "$subdir/run.log" | grep -oP '[\d.]+' | tail -1)
    echo "  rc=$rc tp=${tp:-FAIL} acc=${acc:-n/a}"
    if [ $rc -ne 0 ] || [ -z "$tp" ]; then
        grep -E "probability|Traceback|RuntimeError|OutOfMemory" "$subdir/run.log" | head -3
    fi
}

# 34B int4 async spec sampling (the real target scenario) — TP=4 target + 1 draft
# CodeLlama-34B is fp16 native → bf16 upcast → int4
run "1_34b_int4_async" --llama --size 8 --model_path "$TARGET" --gpus 5 --b 1 --temp 0.6 \
    --output_len 48 --max_model_len 1024 --all --numseqs 2 \
    --draft_path "$DRAFT" --async --spec --k 7 --flh 5 4 4 3 3 2 2 1 --quant_int4

# 34B int4 MESA (with no_quant_lm_head)
run "2_34b_int4_mesa" --llama --size 8 --model_path "$TARGET" --gpus 5 --b 1 --temp 0.6 \
    --output_len 48 --max_model_len 1024 --all --numseqs 2 \
    --draft_path "$DRAFT" --async --spec --k 5 --f 4 \
    --mesa --mesa_exit_layer 32 --mesa_draft_fan_out 2 \
    --quant_int4 --no_quant_lm_head

echo ""
echo "=== SUMMARY ==="
for t in 1_34b_int4_async 2_34b_int4_mesa; do
    log="$OUT/$t/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    mem=$(grep -oP 'bf16_bytes=\K[\d.]+' "$log" 2>/dev/null | head -1)
    printf "%-30s TP=%-8s accept=%-8s\n" "$t" "${tp:-FAIL}" "${acc:-n/a}"
done
