#!/bin/bash
# Final experiment: AR / Baseline SSD / MESA x2 exits
# Target: layerskip-codellama-34B (TP=4)
# Draft:  TinyLlama-1.1B (TP=1)
# Prompts: 4 datasets × 50 = 200 (humaneval, alpaca, gsm8k, ultrafeedback)
# K=6, front-loaded geometric fanout --flh 5 4 4 3 2 2 1 (MQ_LEN=21)
# MESA exit: 24 (1/2 of 48), 32 (2/3 of 48). MESA uses uniform f=3, dfo=1 (MQ_LEN=21, matched)

set -u
cd "$(dirname "$0")/../.."   # ssd repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12290
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_PROFILE_MESA=1

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=$(ls -d /data2/chokwans99/models/models--facebook--layerskip-codellama-34B/snapshots/*/ | head -1)
TARGET="${TARGET%/}"
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
OUT=/home/chokwans99/PSD/ssd/tmp/final_exp

# Dataset: --all picks 50 each from humaneval/alpaca/gsm/ultrafeedback (config in bench_helpers.py:203)
# Total prompts = 4 × 50 = 200
COMMON="--llama --size 8 --model_path $TARGET --gpus 5 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""; echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 1
    # Profile dir per-run
    SSD_PROFILE_DIR="$subdir" \
      "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  -> TP=${tp:-FAIL}"
}

# 1) AR — target only (TP=4, no draft). spec_wait labels won't fire since no speculator
#    Use --gpus 4 because no draft needed; avoids TP=5 weight load problem
SSD_PROFILE_MESA=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$PY" -O bench/bench.py --llama --size 8 --model_path "$TARGET" \
    --gpus 4 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50 \
    > "$OUT/ar/run.log" 2>&1
ar_tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/ar/run.log" | head -1)
echo "AR TP: ${ar_tp:-FAIL}"

# 2) Baseline SSD — K=6, geometric fanout [5,4,4,3,2,2,1] sum=21
run "baseline" --draft_path "$DRAFT" --async --spec --k 6 --flh 5 4 4 3 2 2 1

# 3) MESA exit=24 (1/2) — K=6, uniform f=3 dfo=1 (MQ_LEN=21, matched to baseline)
run "mesa_exit24" --draft_path "$DRAFT" --async --spec --k 6 --f 3 \
    --mesa --mesa_exit_layer 24 --mesa_draft_fan_out 1

# 4) MESA exit=32 (2/3) — same knobs except exit_layer
run "mesa_exit32" --draft_path "$DRAFT" --async --spec --k 6 --f 3 \
    --mesa --mesa_exit_layer 32 --mesa_draft_fan_out 1

echo ""
echo "=== SUMMARY ==="
printf "%-18s %-10s %-8s %-8s %-8s %-10s %-10s\n" "tag" "TP" "CH" "TS" "Accept" "Draft_ms" "Verify_ms"
for f in ar baseline mesa_exit24 mesa_exit32; do
    log="$OUT/$f/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    ch=$(grep "Avg Cache Hits" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ts=$(grep "Avg Tokens per step" "$log" 2>/dev/null | head -1 | grep -oP '[\d.]+' | tail -1)
    ar=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ds=$(grep "Avg draft step" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    tv=$(grep "Avg target verify" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "%-18s %-10s %-8s %-8s %-8s %-10s %-10s\n" "$f" "${tp:-?}" "${ch:-?}" "${ts:-?}" "${ar:-?}" "${ds:-?}" "${tv:-?}"
done | tee "$OUT/SUMMARY.txt"
