#!/bin/bash
# T10 — Hybrid vs Optimized Split (corrected K1/K2) on 70B both-AWQ.
# Predictor-driven candidates: top-1 (K1, K2) per fanout combo from
# experiments/hybrid_timing_sweep/predictor.py output. Run BOTH modes at the
# same (K1, K2) for apples-to-apples timing breakdown comparison.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export SSD_PROFILE_MESA=1

N_GPUS_NEEDED=5
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }' \
        | head -n $N_GPUS_NEEDED | paste -sd,
    )
    n_free=$(echo "$free_gpus" | tr ',' '\n' | grep -c .)
    if [ "$n_free" -lt "$N_GPUS_NEEDED" ]; then
        echo "ERROR: only $n_free free GPUs found." >&2; exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] using free GPUs: $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="${1:-$PWD/experiments/hybrid_vs_split_70b_v2/results}"
mkdir -p "$OUT"
echo "Output: $OUT"

# Same short-run setup — 10 prompts × output_len=64 (~3 min/run)
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run() {
    local tag="$1"; shift
    local mode="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    if [ -f "$subdir/run.log" ] && grep -q "Total Throughput" "$subdir/run.log"; then
        echo "[skip] $tag (already done)"; return
    fi
    echo "[$(date +%H:%M:%S)] === $tag (mode=$mode) === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
    local force_split=0
    [ "$mode" = "force_split" ] && force_split=1
    SSD_PROFILE_DIR="$subdir" SSD_FORCE_SPLIT_PHASE2="$force_split" \
        "$PY" -O bench/bench.py $COMMON $QUANT "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?}"; sleep 3
}

# 5 predictor-driven (dfo, pfo, K1, K2) × 2 modes = 10 runs
# Predictor table:
#   dfo=2, pfo=3 → K1=9, K2=1   (low fanout, deep K1)
#   dfo=3, pfo=3 → K1=8, K2=2   (balanced, lowest predicted idle)
#   dfo=3, pfo=5 → K1=8, K2=2   (high pfo, large fanout asymmetry)
#   dfo=4, pfo=4 → K1=7, K2=2   (dfo=4 best)
#   dfo=4, pfo=6 → K1=7, K2=2   (largest fanout)

run "dfo2_pfo3_K10_K1_9_K2_1_hybrid" default     --k 10 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 1
run "dfo2_pfo3_K10_K1_9_K2_1_split"  force_split --k 10 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 1

run "dfo3_pfo3_K10_K1_8_K2_2_hybrid" default     --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo3_pfo3_K10_K1_8_K2_2_split"  force_split --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2

run "dfo3_pfo5_K10_K1_8_K2_2_hybrid" default     --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo3_pfo5_K10_K1_8_K2_2_split"  force_split --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2

run "dfo4_pfo4_K9_K1_7_K2_2_hybrid"  default     --k 9 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 2
run "dfo4_pfo4_K9_K1_7_K2_2_split"   force_split --k 9 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 2

run "dfo4_pfo6_K9_K1_7_K2_2_hybrid"  default     --k 9 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 2
run "dfo4_pfo6_K9_K1_7_K2_2_split"   force_split --k 9 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 2

echo ""
echo "===== plots + summary ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue; [ -f "$d/run.log" ] || continue
    grep -q "Total Throughput" "$d/run.log" || continue
    "$PY" bench/plot_mesa_timeline.py "$d" --step 50 --warmup 5 >/dev/null 2>&1
    "$PY" bench/plot_mesa_breakdown.py "$d" >/dev/null 2>&1
done

echo ""
echo "===== compare table (predictor-driven candidates) ====="
{
printf "%-40s %-10s %-8s %-8s %-8s %-10s %-10s\n" "config" "TPS" "accept" "P1" "P2" "draft_ms" "verify_ms"
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    log="$d/run.log"; [ -f "$log" ] || continue
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" | head -1)
    ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$log" | head -1)
    p1=$(grep -oP 'Avg Phase 1 \(draft\) Hit Rate:\s*\K[\d.]+' "$log" | head -1)
    p2=$(grep -oP 'Avg Phase 2 \(proxy\) Hit Rate:\s*\K[\d.]+' "$log" | head -1)
    dms=$(grep -oP 'Avg draft step time \(ms\):\s*\K[\d.]+' "$log" | head -1)
    vms=$(grep -oP 'Avg target verify time \(ms\):\s*\K[\d.]+' "$log" | head -1)
    printf "%-40s %-10s %-8s %-8s %-8s %-10s %-10s\n" \
        "$label" "${tp:-?}" "${ar:-?}" "${p1:-?}" "${p2:-?}" "${dms:-?}" "${vms:-?}"
done
} | tee "$OUT/SUMMARY.txt"
