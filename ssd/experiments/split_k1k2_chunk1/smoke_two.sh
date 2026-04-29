#!/bin/bash
# Two configurations: K1==K2 and K1>K2.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12397

N_GPUS_NEEDED=3
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }' \
        | head -n $N_GPUS_NEEDED | paste -sd,
    )
    n_free=$(echo "$free_gpus" | tr ',' '\n' | grep -c .)
    [ "$n_free" -lt "$N_GPUS_NEEDED" ] && { echo "ERROR: only $n_free GPUs"; exit 1; }
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT="$PWD/experiments/split_k1k2_chunk1"

case_run() {
    local tag=$1; local K1=$2; local K2=$3
    local K=$((K1 + K2))
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
    SSD_FORCE_SPLIT_K1K2=1 "$PY" -O bench/bench.py \
        --llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec \
        --gpus 3 --b 1 --temp 0 --output_len 32 --max_model_len 2048 --random --numseqs 3 \
        --k $K --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 \
        --mesa_phase1_k $K1 --mesa_phase2_k $K2 \
        >"$OUT/$tag.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/$tag.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|Aborted|FAIL' "$OUT/$tag.log" | head -1)
    echo "  $tag (K1=$K1, K2=$K2): rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    grep -E "MESA split-K1K2.*layouts|MESA split-K1K2.*Captured" "$OUT/$tag.log" | head -3
    sleep 2
}

case_run "test_K1eqK2" 3 3
case_run "test_K2ltK1" 3 2
