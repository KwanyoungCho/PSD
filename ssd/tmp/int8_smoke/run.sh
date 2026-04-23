#!/bin/bash
# INT8 weight-only smoke test: regression + quant paths.
# Uses layerskip-llama2-7B TP=2 + TinyLlama draft.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12297
export CUDA_VISIBLE_DEVICES=0,1,2
export SSD_PROFILE_MESA=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/models/layerskip-llama2-7B
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
OUT=/home/chokwans99/PSD/ssd/tmp/int8_smoke

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

# 1) Baseline SSD (flag off) — must match pre-quant
run "baseline_noquant" --draft_path "$DRAFT" --async --spec --k 7 --flh 5 4 4 3 3 2 2 1 || exit 1

# 2) Baseline SSD + INT8 quant (target only)
run "baseline_int8" --draft_path "$DRAFT" --async --spec --k 7 --flh 5 4 4 3 3 2 2 1 --quant_int8 || exit 1

# 3) MESA (no quant) — regression
run "mesa_noquant" --draft_path "$DRAFT" --async --spec --k 5 --f 4 \
    --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2 || exit 1

# 4) MESA + INT8 (lm_head on, default)
run "mesa_int8_lmhead_on" --draft_path "$DRAFT" --async --spec --k 5 --f 4 \
    --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2 --quant_int8 || exit 1

# 5) MESA + INT8 (lm_head OFF for MESA proxy quality)
run "mesa_int8_lmhead_off" --draft_path "$DRAFT" --async --spec --k 5 --f 4 \
    --mesa --mesa_exit_layer 20 --mesa_draft_fan_out 2 --quant_int8 --no_quant_lm_head || exit 1

echo ""
echo "=== SMOKE PASS ==="
