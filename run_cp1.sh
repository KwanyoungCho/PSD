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

QWEN_DONE_MARKER="$OUT_DIR/correction_Qwen3-32B.json"

# ── Qwen 실험 완료 대기 ──
echo "Waiting for Qwen experiment to finish..."
while [ ! -f "$QWEN_DONE_MARKER" ]; do
    echo "  $(date '+%H:%M:%S')  Qwen not done yet, checking in 60s..."
    sleep 60
done
echo "$(date '+%H:%M:%S')  Qwen done! Starting cp1 experiments."

LOG_FILE="$OUT_DIR/run_cp1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT_DIR"
echo "Logging to: $LOG_FILE"

run_exp() {
    local name="$1" target="$2" draft="$3"
    echo ""
    echo "========================================"
    echo "  $name  [cp0, cp1 only]"
    echo "  target: $target"
    echo "  draft:  $draft"
    echo "========================================"
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT" \
        --target      "$target" \
        --draft       "$draft" \
        --n_samples   "$N_SAMPLES" \
        --output_dir  "$OUT_DIR" \
        --cache_dir   "$CACHE_DIR" \
        --checkpoints 0 1 \
        --suffix      "_cp01"
}

{
    run_exp "Llama 70B + 1B  [cp01]"   "$LLAMA_TARGET" "$LLAMA_DRAFT"
    run_exp "Qwen3 32B + 0.6B [cp01]"  "$QWEN_TARGET"  "$QWEN_DRAFT"

    echo ""
    echo "All cp1 experiments done. Results in $OUT_DIR"
} 2>&1 | tee "$LOG_FILE"
