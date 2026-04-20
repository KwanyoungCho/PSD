#!/bin/bash
# Re-run MESA only (baseline/AR results were valid — single _decode_tree call per step)
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
    rm -f "$subdir"/mesa_profile_*.json "$subdir"/run.log 2>/dev/null
    mkdir -p "$subdir"
    echo ""; echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    SSD_PROFILE_DIR="$subdir" \
      "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  -> TP=${tp:-FAIL}"
}

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
