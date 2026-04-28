#!/bin/bash
# hybrid_vs_split_70b — head-to-head comparison of MESA Phase 2 hybrid (default)
# vs split fallback at the known optimum (K=5, K1=3, K2=2, exit=40, dfo=2, f=4).
#
# Matches `tmp/final_exp2_quant_70b` setup so split numbers cross-validate
# against the existing baseline report (mesa_k5_f4_dfo2_exit40 → 61.02 tok/s).
#
# SSD_PROFILE_MESA=1 captures per-phase CUDA-event timings for both runs.
# Profiler uses zero-sync CUDA events with early-return None when off,
# so the labeled hot-path adds no measurable overhead.

set -u
cd "$(dirname "$0")/../.."   # ssd repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12297
export SSD_PROFILE_MESA=1

# --- Auto-pick free GPUs ---
# Need 5 GPUs (4 target TP + 1 draft). Prefer GPUs that are currently idle
# (mem.used < 1GiB AND util < 5%) so we don't fight other users.
# Override by exporting CUDA_VISIBLE_DEVICES before invoking this script.
N_GPUS_NEEDED=5
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v N=$N_GPUS_NEEDED '
            { gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }
        ' | head -n $N_GPUS_NEEDED | paste -sd,
    )
    n_free=$(echo "$free_gpus" | tr ',' '\n' | grep -c .)
    if [ "$n_free" -lt "$N_GPUS_NEEDED" ]; then
        echo "WARN: only $n_free free GPUs found (need $N_GPUS_NEEDED). Falling back to 0..4." >&2
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4
    else
        export CUDA_VISIBLE_DEVICES="$free_gpus"
        echo "[gpu-pick] using free GPUs: $CUDA_VISIBLE_DEVICES"
    fi
else
    echo "[gpu-pick] CUDA_VISIBLE_DEVICES override: $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="${1:-$PWD/experiments/hybrid_vs_split_70b/results}"
mkdir -p "$OUT"
echo "Output: $OUT"

# Default for new experiments: BOTH target and draft AWQ-quantized.
# Pre-hybrid both-AWQ baseline lives at experiments/quant_awq/70b/draft_awq/
# for direct A/B/C comparison across hybrid restructuring.
# Prompt set matches: 200 prompts (4 datasets × 50), output_len=256.
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"
MESA_BASE="--mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --k 5 --f 4"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    if [ -f "$subdir/run.log" ] && grep -q "Total Throughput" "$subdir/run.log"; then
        echo "[skip] $tag (already done)"
        return
    fi
    echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true
    sleep 2
    SSD_PROFILE_DIR="$subdir" "$PY" -O bench/bench.py $COMMON $QUANT "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local ch=$(grep -oP 'Avg Cache Hits:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} accept=${ar:-?} cache_hits=${ch:-?}"
    sleep 3
}

# --- 1) Split fallback (legacy two-pass MESA, K=5/dfo=2/exit=40) ---
# SSD_FORCE_SPLIT_PHASE2=1 forces split path even when K1/K2 set; here we just
# omit K1/K2 for a cleaner config (= legacy path) so it matches
# tmp/final_exp2_quant_70b/mesa_k5_f4_dfo2_exit40 1:1.
run "split_k5_dfo2_exit40" $MESA_BASE

# --- 2) Hybrid optimum (K1=3, K2=2, exit=40, dfo=2, f=4) ---
run "hybrid_k5_K1_3_K2_2_exit40" $MESA_BASE --mesa_phase1_k 3 --mesa_phase2_k 2

echo ""
echo "===== SUMMARY (top-level metrics) ====="
"$PY" bench/extract_sweep_metrics.py "$OUT" | tee "$OUT/SUMMARY.md"

echo ""
echo "===== Profile artifacts ====="
for d in "$OUT"/*/; do
    label=$(basename "$d")
    n_t=$(ls "$d"/mesa_profile_target_rank0_*.json 2>/dev/null | wc -l)
    n_d=$(ls "$d"/mesa_profile_draft_*.json 2>/dev/null | wc -l)
    echo "  $label: target_rank0=$n_t  draft=$n_d"
done

echo ""
echo "Next: post-process per-phase breakdown:"
echo "  python bench/plot_mesa_breakdown.py $OUT/split_k5_dfo2_exit40"
echo "  python bench/plot_mesa_breakdown.py $OUT/hybrid_k5_K1_3_K2_2_exit40"
echo "  python experiments/hybrid_vs_split_70b/compare_breakdown.py $OUT"
