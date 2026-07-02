#!/bin/bash
# CodeLlama-34B + TinyLlama-1.1B — DUET vs baseline sweep
# Target TP=4, Draft TP=1, total 5 GPUs (0..4), GPUs 5..7 unused.
# Run serially (single 5-GPU job at a time).

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/tmp
export SSD_DIST_PORT=12280
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python

TARGET=$(ls -d /data2/chokwans99/models/models--facebook--layerskip-codellama-34B/snapshots/*/ 2>/dev/null | head -1)
TARGET="${TARGET%/}"
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0

if [ -z "$TARGET" ] || [ ! -f "$TARGET/config.json" ]; then
    echo "ERROR: target model not found. Check /data2/chokwans99/models/models--facebook--layerskip-codellama-34B"
    exit 1
fi

OUTDIR="${1:-/home/chokwans99/PSD/ssd/tmp/34b_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"
echo "Target: $TARGET"
echo "Draft:  $DRAFT"
echo "Out:    $OUTDIR"

run() {
    local label="$1"; shift
    local outfile="$OUTDIR/${label}.log"
    echo ""
    echo "[$(date +%H:%M:%S)] === $label === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 1
    "$PY" -O bench/bench.py --llama --size 8 \
        --model_path "$TARGET" --draft_path "$DRAFT" \
        --gpus 5 --b 1 --temp 0.6 --numseqs 30 --output_len 256 --random \
        --max_model_len 2048 \
        "$@" >"$outfile" 2>&1
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$outfile" | head -1)
    echo "  -> TP=${tp:-FAIL}"
}

# === Phase 0: AR baseline (no spec) — establish lower bound ===
run "ar_34b"

# === Phase 1: sync speculation baselines for f sweep ===
# we use --async so timing matches DUET pipeline; baseline varies --f
for f in 2 3 4 5 6 8; do
    run "baseline_f${f}" --async --spec --k 4 --f ${f}
done

# === Phase 2: DUET at each f, dfo=1 (phase1=1, phase2=f-1) — matches best 8B config ===
for f in 2 3 4 5 6 8; do
    run "duet_f${f}_dfo1" --async --spec --k 4 --f ${f} \
        --duet --duet_exit_layer 32 --duet_draft_fan_out 1
done

echo ""
echo "=== SUMMARY ==="
printf "%-22s %-10s %-8s %-8s %-10s %-10s\n" "label" "TP" "CH" "TS" "Draft" "Verify"
for f in "$OUTDIR"/*.log; do
    label=$(basename "$f" .log)
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$f" 2>/dev/null | head -1)
    ch=$(grep "Avg Cache Hits" "$f" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ts=$(grep "Avg Tokens per step" "$f" 2>/dev/null | head -1 | grep -oP '[\d.]+' | tail -1)
    ds=$(grep "Avg draft step" "$f" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    tv=$(grep "Avg target verify" "$f" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "%-22s %-10s %-8s %-8s %-10s %-10s\n" "$label" "${tp:-?}" "${ch:-?}" "${ts:-?}" "${ds:-?}ms" "${tv:-?}ms"
done | tee "$OUTDIR/SUMMARY.txt"
