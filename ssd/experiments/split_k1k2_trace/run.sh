#!/bin/bash
# Contract validation traces for split-K1/K2 path.
# 4 minimal configs × 1 prompt × temp=0 × short output.
# Logs: hit_cache, spec→verify, proxy_compute, proxy_unpack, proxy_seed.
# Goal: prove which positions of proxy/seed are real vs zero-pad.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12397
export SSD_FORCE_SPLIT_K1K2=1
export SSD_TRACE_SPLIT_K1K2=1

N_GPUS_NEEDED=3
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }' \
        | head -n $N_GPUS_NEEDED | paste -sd,
    )
    n_free=$(echo "$free_gpus" | tr ',' '\n' | grep -c .)
    [ "$n_free" -lt "$N_GPUS_NEEDED" ] && { echo "ERROR: only $n_free free GPUs found." >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] using free GPUs: $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT="$PWD/experiments/split_k1k2_trace"
mkdir -p "$OUT"

# Short: 1 prompt, output_len=8 (≈ 2-4 spec steps), temp=0.
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 3 --b 1 --temp 0 --output_len 8 --max_model_len 2048 --random --numseqs 1"

run_case() {
    local name="$1"; local K1="$2"; local K2="$3"
    local K=$((K1 + K2))
    local F=$((2 + 2))   # dfo=2 + pfo=2
    local subdir="$OUT/$name"
    mkdir -p "$subdir"
    echo "[$(date +%H:%M:%S)] === $name (K1=$K1, K2=$K2, K=$K) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
    "$PY" -O bench/bench.py $COMMON --k $K --f $F --mesa --mesa_exit_layer 21 \
        --mesa_draft_fan_out 2 --mesa_phase1_k $K1 --mesa_phase2_k $K2 \
        >"$subdir/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|Aborted' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 2
}

run_case "K1_3_K2_1" 3 1
run_case "K1_2_K2_2" 2 2
run_case "K1_2_K2_3" 2 3
run_case "K1_1_K2_3" 1 3

echo ""
echo "===== trace summaries ====="
for d in "$OUT"/K1_*/; do
    name=$(basename "$d")
    log="$d/run.log"
    [ -f "$log" ] || continue
    echo ""
    echo "--- $name ---"
    grep -E "TRACE-split-k1k2 #[1-4]" "$log" | head -25
done
