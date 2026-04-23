#!/bin/bash
# Phase 3: verify CUDA graph capture/replay with int4 across decode/verify/speculate paths
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
OUT=/home/chokwans99/PSD/ssd/tmp/int4_phase3

pick() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 14000 {print $1}' | head -$1 | paste -sd,; }

GPUS=$(pick 3)
[ -z "$GPUS" ] && { echo "no GPUs"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPUS

C_TO="--llama --size 8 --model_path $TARGET --gpus 2 --b 1 --output_len 48 --max_model_len 512 --all --numseqs 2 --temp 0.6"
C_SP="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --output_len 48 --max_model_len 512 --all --numseqs 2 --temp 0.6 --draft_path $DRAFT --async --spec --k 7 --flh 5 4 4 3 3 2 2 1"
C_MS="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --output_len 48 --max_model_len 512 --all --numseqs 2 --temp 0.6 --draft_path $DRAFT --async --spec --k 5 --f 4 --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2"

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
    if [ $rc -ne 0 ] || [ -z "$tp" ]; then
        echo "  -> FAIL (rc=$rc)"
        grep -E "probability|Traceback|RuntimeError|Error:" "$subdir/run.log" | head -3
        return 1
    fi
    echo "  -> OK  TP=$tp  accept=${acc:-n/a}"
}

# CUDA graph (default) vs --eager, int4, all three paths
run "1_ar_int4_graph"         $C_TO --quant_int4
run "2_ar_int4_eager"         $C_TO --quant_int4 --eager
run "3_spec_int4_graph"       $C_SP --quant_int4
run "4_spec_int4_eager"       $C_SP --quant_int4 --eager
run "5_mesa_int4_graph"       $C_MS --quant_int4 --no_quant_lm_head
run "6_mesa_int4_eager"       $C_MS --quant_int4 --no_quant_lm_head --eager

echo ""
echo "=== SUMMARY ==="
printf "%-30s %-10s %-10s\n" "config" "TP" "accept"
for t in 1_ar_int4_graph 2_ar_int4_eager 3_spec_int4_graph 4_spec_int4_eager 5_mesa_int4_graph 6_mesa_int4_eager; do
    log="$OUT/$t/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "%-30s %-10s %-10s\n" "$t" "${tp:-FAIL}" "${acc:-n/a}"
done
