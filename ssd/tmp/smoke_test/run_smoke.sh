#!/bin/bash
# Minimal smoke test: verify MESA + baseline still run after dead code removal.
# Uses layerskip-llama2-7B (smaller than 34B so we fit around other jobs on the GPUs).
# 32 layers -> exit_layer=20. TP=2 target + TP=1 draft = 3 GPUs.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12296
export CUDA_VISIBLE_DEVICES=0,3,6
export SSD_PROFILE_MESA=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama2-7B
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
OUT=/home/chokwans99/PSD/ssd/tmp/smoke_test

COMMON="--llama --size 8 --model_path $TARGET --gpus 3 --b 1 --temp 0.6 --output_len 48 --max_model_len 512 --all --numseqs 2"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""; echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 1
    SSD_PROFILE_DIR="$subdir" \
      "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    if [ $rc -ne 0 ] || [ -z "$tp" ]; then
        echo "  -> FAIL (rc=$rc). Root error:"
        grep -E "(OutOfMemory|AssertionError|RuntimeError|TypeError|NameError|ImportError|KeyError|ValueError|IndexError)" "$subdir/run.log" | head -5
        echo "  tail:"
        tail -15 "$subdir/run.log"
        return 1
    fi
    echo "  -> OK TP=$tp"
}

# 1) Baseline SSD K=7 geo (flh must sum to MQ_LEN=24)
run "baseline_k7_geo" --draft_path "$DRAFT" --async --spec --k 7 --flh 5 4 4 3 3 2 2 1 || exit 1

# 2) MESA K=5 f=4 dfo=2 exit=20 (32-layer model)
run "mesa_k5_f4_dfo2_exit20" --draft_path "$DRAFT" --async --spec --k 5 --f 4 \
    --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2 || exit 1

echo ""
echo "=== SMOKE TEST PASS ==="
