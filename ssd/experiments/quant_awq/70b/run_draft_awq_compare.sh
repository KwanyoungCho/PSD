#!/bin/bash
# Draft AWQ perf comparison on layerskip-llama2-70B target.
#
# Compares for each spec config:
#   A. target AWQ + draft DENSE   (final_exp2_quant_70b/{config}/run.log)
#   B. target AWQ + draft AWQ     (this script's runs, written under draft_awq/)

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12299
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_PROFILE_MESA=1

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
OUT=/home/chokwans99/PSD/ssd/tmp/final_exp2_quant_70b/draft_awq
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
mkdir -p "$OUT"

COMMON="--llama --size 8 --model_path $TARGET --gpus 5 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50"
QUANT_TGT="--quant_awq --quant_awq_artifact $TGT_ART"
QUANT_DRAFT="--quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    SSD_PROFILE_DIR="$subdir" "$PY" -O bench/bench.py $COMMON $QUANT_TGT $QUANT_DRAFT \
      --draft_path "$DRAFT" "$@" >"$subdir/run.log" 2>&1
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  -> TP=${tp:-FAIL}"
    sleep 3
}

# No AR here — AR has no draft.
run "baseline_k7_uniform"  --async --spec --k 7 --f 3
run "baseline_k7_geo"      --async --spec --k 7 --flh 5 4 4 3 3 2 2 1
for EX in 40 47 53; do
    run "mesa_k5_f4_dfo2_exit${EX}"  --async --spec --k 5 --f 4 --mesa \
        --mesa_exit_layer $EX --mesa_draft_fan_out 2
done

echo ""
echo "=== SUMMARY (target AWQ + draft AWQ, 70B) ==="
printf "%-28s %-10s %-8s %-8s %-8s %-10s %-10s\n" "tag" "TP" "CH" "TS" "Accept" "Draft_ms" "Verify_ms" | tee "$OUT/SUMMARY.txt"
for f in baseline_k7_uniform baseline_k7_geo mesa_k5_f4_dfo2_exit40 mesa_k5_f4_dfo2_exit47 mesa_k5_f4_dfo2_exit53; do
    log="$OUT/$f/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    ch=$(grep "Avg Cache Hits" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ts=$(grep "Avg Tokens per step" "$log" 2>/dev/null | head -1 | grep -oP '[\d.]+' | tail -1)
    ar=$(grep "Avg Fraction" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    ds=$(grep "Avg draft step" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    tv=$(grep "Avg target verify" "$log" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    printf "%-28s %-10s %-8s %-8s %-8s %-10s %-10s\n" "$f" "${tp:-?}" "${ch:-?}" "${ts:-?}" "${ar:-?}" "${ds:-?}" "${tv:-?}" | tee -a "$OUT/SUMMARY.txt"
done
