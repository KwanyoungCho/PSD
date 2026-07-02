#!/bin/bash
# Phase 9B hybrid sweep: layerskip-llama2-70B (AWQ TP=4) + TinyLlama-1.1B (TP=1).
# Compares baseline split DUET vs Phase 9B hybrid configs.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/tmp

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
QUANT="--quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 --quant_group_size 128"

OUT="${1:-/tmp/hybrid_sweep_70b_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "Output: $OUT"

NUMSEQS=50
OUTLEN=128
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len $OUTLEN --max_model_len 2048 --numseqs $NUMSEQS --random $QUANT"

run() {
    local label="$1"; shift
    local outdir="$OUT/$label"
    mkdir -p "$outdir"
    if [ -f "$outdir/run.log" ] && grep -q "Total Throughput" "$outdir/run.log"; then
        echo "[skip] $label (already done)"
        return
    fi
    echo "===== $label ====="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_PROFILE_DUET=0 \
        "$PY" -O bench/bench.py $COMMON "$@" >"$outdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$outdir/run.log" | head -1)
    local ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$outdir/run.log" | head -1)
    local ch=$(grep -oP 'Avg Cache Hits:\s*\K[\d.]+' "$outdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} accept=${ar:-?} cache_hits=${ch:-?}"
}

# Baseline split DUET (existing best from final_exp2_quant_70b: K=5 dfo=2 exit=40)
run "split_k5_dfo2_exit40" --k 5 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2

# Phase 9B hybrid sweep (K = K1 + K2 = speculate_k)
# K=5 odd splits
run "hybrid_k5_K1_2_K2_3_exit40" --k 5 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2 --duet_phase1_k 2 --duet_phase2_k 3
run "hybrid_k5_K1_3_K2_2_exit40" --k 5 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2 --duet_phase1_k 3 --duet_phase2_k 2
# K=6 balanced
run "hybrid_k6_K1_3_K2_3_exit40" --k 6 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2 --duet_phase1_k 3 --duet_phase2_k 3
run "hybrid_k6_K1_3_K2_3_exit47" --k 6 --f 4 --duet --duet_exit_layer 47 --duet_draft_fan_out 2 --duet_phase1_k 3 --duet_phase2_k 3
# K=8 balanced (validation config from 8B smoke)
run "hybrid_k8_K1_4_K2_4_exit40" --k 8 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2 --duet_phase1_k 4 --duet_phase2_k 4
run "hybrid_k8_K1_4_K2_4_exit47" --k 8 --f 4 --duet --duet_exit_layer 47 --duet_draft_fan_out 2 --duet_phase1_k 4 --duet_phase2_k 4
# Asymmetric K=8
run "hybrid_k8_K1_2_K2_6_exit40" --k 8 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2 --duet_phase1_k 2 --duet_phase2_k 6
run "hybrid_k8_K1_6_K2_2_exit40" --k 8 --f 4 --duet --duet_exit_layer 40 --duet_draft_fan_out 2 --duet_phase1_k 6 --duet_phase2_k 2

echo ""
echo "===== SUMMARY ====="
printf "%-32s %-10s %-8s %-8s\n" "config" "TP" "accept" "cache_hits"
for d in "$OUT"/*/; do
    label=$(basename "$d")
    log="$d/run.log"
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    ch=$(grep -oP 'Avg Cache Hits:\s*\K[\d.]+' "$log" 2>/dev/null | head -1)
    printf "%-32s %-10s %-8s %-8s\n" "$label" "${tp:-?}" "${ar:-?}" "${ch:-?}"
done | tee "$OUT/SUMMARY.txt"
