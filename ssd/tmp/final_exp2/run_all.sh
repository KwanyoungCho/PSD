#!/bin/bash
# Rev1 final experiment: AR / Baseline SSD (K=7 uniform + geo) / MESA (K=5, f=4, dfo=2, exit 24/28/32)
# Target: layerskip-codellama-34B TP=4. Draft: TinyLlama-1.1B TP=1.
# Post-Rev1 fixes: #3 (B=1 assert), #4 (proxy_top_k 확대 + fallback 제거), #D (spec buffer prealloc), #1 (벡터화), #8 (dead code).

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12295
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_PROFILE_MESA=1

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=$(ls -d /data2/chokwans99/models/models--facebook--layerskip-codellama-34B/snapshots/*/ | head -1)
TARGET="${TARGET%/}"
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
OUT=/home/chokwans99/PSD/ssd/tmp/final_exp2

COMMON="--llama --size 8 --model_path $TARGET --gpus 5 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo ""; echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 1
    SSD_PROFILE_DIR="$subdir" \
      "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  -> TP=${tp:-FAIL}"
}

# 1) AR — target only TP=4 (no draft, no spec_wait labels)
mkdir -p "$OUT/ar"
SSD_PROFILE_MESA=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$PY" -O bench/bench.py --llama --size 8 --model_path "$TARGET" \
    --gpus 4 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50 \
    > "$OUT/ar/run.log" 2>&1
ar_tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/ar/run.log" | head -1)
echo "AR TP: ${ar_tp:-FAIL}"

# 2) Baseline SSD K=7, uniform f=3 (MQ_LEN=24)
run "baseline_k7_uniform" --draft_path "$DRAFT" --async --spec --k 7 --f 3

# 3) Baseline SSD K=7, geometric [5,4,4,3,3,2,2,1] sum=24
run "baseline_k7_geo" --draft_path "$DRAFT" --async --spec --k 7 --flh 5 4 4 3 3 2 2 1

# 4) MESA K=5 f=4 dfo=2 (phase1=2, phase2=2 — 1:1 split), exit 24/28/32
for EX in 24 28 32; do
    run "mesa_k5_f4_dfo2_exit${EX}" --draft_path "$DRAFT" --async --spec --k 5 --f 4 \
        --mesa --mesa_exit_layer $EX --mesa_draft_fan_out 2
done

echo ""
echo "=== SUMMARY ==="
printf "%-28s %-10s %-8s %-8s %-8s %-10s %-10s\n" "tag" "TP" "CH" "TS" "Accept" "Draft_ms" "Verify_ms"
for f in ar baseline_k7_uniform baseline_k7_geo mesa_k5_f4_dfo2_exit24 mesa_k5_f4_dfo2_exit28 mesa_k5_f4_dfo2_exit32; do
    log="$OUT/$f/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    ch=$(grep "Avg Cache Hits" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ts=$(grep "Avg Tokens per step" "$log" 2>/dev/null | head -1 | grep -oP '[\d.]+' | tail -1)
    ar=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ds=$(grep "Avg draft step" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    tv=$(grep "Avg target verify" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "%-28s %-10s %-8s %-8s %-8s %-10s %-10s\n" "$f" "${tp:-?}" "${ch:-?}" "${ts:-?}" "${ar:-?}" "${ds:-?}" "${tv:-?}"
done | tee "$OUT/SUMMARY.txt"
