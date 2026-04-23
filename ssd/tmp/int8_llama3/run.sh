#!/bin/bash
# Full validation with Llama-3-8B (bf16 native) to test whether fp16 overflow
# hypothesis was correct.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT=/home/chokwans99/PSD/ssd/tmp/int8_llama3

# Pick 3 free GPUs dynamically
pick_gpus() {
    local n=$1
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' '$2 > 14000 {print $1}' | head -$n | paste -sd,
}

GPUS=$(pick_gpus 3)
if [ -z "$GPUS" ] || [ $(echo $GPUS | tr ',' '\n' | wc -l) -lt 3 ]; then
    echo "Not enough free GPUs, got: $GPUS"
    exit 1
fi
echo "using GPUs: $GPUS"
export CUDA_VISIBLE_DEVICES=$GPUS

COMMON_TARGET_ONLY="--llama --size 8 --model_path $TARGET --gpus 2 --b 1 --output_len 48 --max_model_len 512 --all --numseqs 2"
COMMON_SPEC="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --output_len 48 --max_model_len 512 --all --numseqs 2 --draft_path $DRAFT --spec --k 7"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""
    echo "[$(date +%H:%M:%S)] === $tag ==="
    echo "  cmd: $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    "$PY" -O bench/bench.py "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local acc=$(grep "Avg Fraction" "$subdir/run.log" | grep -oP '[\d.]+' | tail -1)
    local has_nan=$(grep -c "probability tensor" "$subdir/run.log")
    local has_oom=$(grep -c "CUDA out of memory" "$subdir/run.log")
    if [ $rc -ne 0 ] || [ -z "$tp" ]; then
        echo "  -> FAIL (rc=$rc, nan_assert=$has_nan, oom=$has_oom)"
        return 1
    fi
    echo "  -> OK  TP=$tp  accept=${acc:-n/a}"
}

# 1. Dense flag-off regression — AR
run "1_ar_dense"  $COMMON_TARGET_ONLY --temp 0.6 || true

# 2. INT8 AR
run "2_ar_int8"  $COMMON_TARGET_ONLY --temp 0.6 --quant_int8 || true

# 3. Dense sync spec (temp=0.6 sampling)
run "3_spec_dense_sampling"  $COMMON_SPEC --temp 0.6 || true

# 4. INT8 sync spec greedy (temp=0)
run "4_spec_int8_greedy"  $COMMON_SPEC --temp 0 --quant_int8 || true

# 5. INT8 sync spec sampling (temp=0.6) — the key test
run "5_spec_int8_sampling"  $COMMON_SPEC --temp 0.6 --quant_int8 || true

# 6. INT8 async spec sampling (temp=0.6) — original failure case
run "6_async_int8_sampling"  $COMMON_SPEC --temp 0.6 --async --flh 5 4 4 3 3 2 2 1 --quant_int8 || true

# 7. MESA smoke with int8 (optional)
run "7_mesa_int8_sampling"  $COMMON_SPEC --temp 0.6 --async --f 4 --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2 --k 5 --quant_int8 || true

echo ""
echo "=== SUMMARY ==="
for d in 1_ar_dense 2_ar_int8 3_spec_dense_sampling 4_spec_int8_greedy 5_spec_int8_sampling 6_async_int8_sampling 7_mesa_int8_sampling; do
    log="$OUT/$d/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    acc=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    nan=$(grep -c "probability tensor" "$log" 2>/dev/null)
    oom=$(grep -c "CUDA out of memory" "$log" 2>/dev/null)
    inf=$(grep -c "inf" "$log" 2>/dev/null | head -1)
    printf "  %-30s TP=%-8s accept=%-7s nan_assert=%s oom=%s\n" "$d" "${tp:-FAIL}" "${acc:-n/a}" "$nan" "$oom"
done
