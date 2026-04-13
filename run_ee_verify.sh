#!/bin/bash
set -euo pipefail

CONDA_ENV="PSD"
SCRIPT="ee_verify_analysis.py"
N_SAMPLES=200
OUT_DIR="/home/chokwans99/Parallel_SD/results"
CACHE_DIR="/data2/shared/huggingface_cache"

LLAMA_TARGET="/data2/chokwans99/models/Llama-3.1-70B-Instruct"
LLAMA_DRAFT="/data2/shared/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"

run_exp() {
    local name="$1" target="$2" draft="$3"
    shift 3
    echo ""
    echo "========================================"
    echo "  $name"
    echo "  target: $target"
    echo "  draft:  $draft"
    echo "  n_samples: $N_SAMPLES"
    echo "  checkpoints: $*"
    echo "========================================"
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
        --target "$target" \
        --draft  "$draft" \
        --n_samples "$N_SAMPLES" \
        --output_dir "$OUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --checkpoints "$@"
}

LOG_FILE="$OUT_DIR/ee_verify_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT_DIR"
echo "Logging to: $LOG_FILE"

{
    run_exp "Llama 70B + 1B (cp1)" "$LLAMA_TARGET" "$LLAMA_DRAFT" 1

    echo ""
    echo "All experiments done. Results in $OUT_DIR"
} 2>&1 | tee "$LOG_FILE"
