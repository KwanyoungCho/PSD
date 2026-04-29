#!/bin/bash
# Test 3 cases: uniform K1=K2, uniform K2<K1, non-uniform Phase 1.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12397

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }' \
        | head -3 | paste -sd,
    )
    [ "$(echo $free_gpus | tr ',' '\n' | wc -l)" -lt 3 ] && { echo "ERROR: need 3 free GPUs"; exit 1; }
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT="$PWD/experiments/split_k1k2_chunk1"

run_case() {
    local tag=$1; local K1=$2; local K2=$3; local p1list=$4
    local K=$((K1 + K2))
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    local extra=""
    [ -n "$p1list" ] && extra="--mesa_split_phase1_fan_out_list $p1list"
    SSD_FORCE_SPLIT_K1K2=1 "$PY" -O bench/bench.py \
        --llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec \
        --gpus 3 --b 1 --temp 0 --output_len 32 --max_model_len 2048 --random --numseqs 3 \
        --k $K --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 \
        --mesa_phase1_k $K1 --mesa_phase2_k $K2 $extra \
        >"$OUT/$tag.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/$tag.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|FAIL|Aborted|NotImplemented|ValueError' "$OUT/$tag.log" | head -1)
    echo "  $tag (K1=$K1 K2=$K2 p1=$p1list): rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    grep -E "MESA split-K1K2.*layouts" "$OUT/$tag.log" | head -1
    sleep 2
}

# 1. Uniform K1=K2
case_run() { run_case "$@"; }
case_run "u_K1eqK2" 4 4 ""

# 2. Uniform K2<K1
case_run "u_K2ltK1" 4 2 ""

# 3. Non-uniform Phase 1 (K1=4 → list len 5)
case_run "nu_K1eqK2" 4 4 "4,3,2,1,1"
case_run "nu_K2ltK1" 4 2 "4,3,2,1,1"
