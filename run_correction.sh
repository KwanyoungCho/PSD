#!/bin/bash
set -euo pipefail

CONDA_ENV="PSD"
SCRIPT="correction_analysis.py"
N_SAMPLES=200
OUT_DIR="/home/chokwans99/Parallel_SD/results"
CACHE_DIR="/data2/shared/huggingface_cache"

LLAMA_TARGET="/data2/chokwans99/models/Llama-3.1-70B-Instruct"
LLAMA_DRAFT="/data2/shared/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"

QWEN_TARGET="/data2/chokwans99/models/Qwen3-32B"
QWEN_DRAFT="/data2/chokwans99/models/Qwen3-0.6B"

run_exp() {
    local name="$1" target="$2" draft="$3"
    echo ""
    echo "========================================"
    echo "  $name"
    echo "  target: $target"
    echo "  draft:  $draft"
    echo "  n_samples: $N_SAMPLES"
    echo "========================================"
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
        --target "$target" \
        --draft  "$draft" \
        --n_samples "$N_SAMPLES" \
        --output_dir "$OUT_DIR" \
        --cache_dir "$CACHE_DIR"
}

LOG_FILE="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT_DIR"
echo "Logging to: $LOG_FILE"

{
    run_exp "Llama 70B + 1B"   "$LLAMA_TARGET" "$LLAMA_DRAFT"
    run_exp "Qwen3 32B + 0.6B" "$QWEN_TARGET"  "$QWEN_DRAFT"

    echo ""
    echo "All experiments done. Results in $OUT_DIR"
} 2>&1 | tee "$LOG_FILE"
