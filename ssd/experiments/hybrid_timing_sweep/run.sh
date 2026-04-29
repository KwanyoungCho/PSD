#!/bin/bash
# Hybrid timing-alignment sweep — focus on per-step idle (proxy_wait,
# draft_recv_cmd, target_spec_wait) at cache-hit timing across (dfo, pfo, K1, K2).
#
# Goal: for each (phase1_fanout, phase2_fanout) combo, find K1/K2 candidates
# that align draft Phase 1 finish with proxy arrival (low proxy_wait) and
# Phase 2 finish with target verify end (low draft_recv_cmd).
#
# Short runs (10 prompts × output_len=64 ≈ 3 min/run) — TPS absolute value
# unreliable, but per-step mean event ms is fine for >=100 spec steps.
#
# exit_layer fixed at 40 (1/2 L). Other exits handled in a separate sweep.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export SSD_PROFILE_MESA=1

# Auto-pick free GPUs
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
        echo "ERROR: only $n_free free GPUs found (need $N_GPUS_NEEDED)." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] using free GPUs: $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="${1:-$PWD/experiments/hybrid_timing_sweep/results}"
mkdir -p "$OUT"
echo "Output: $OUT"

# Short run for screening: 10 prompts × output_len=64 → ~150 spec steps
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

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
    echo "  rc=$rc TP=${tp:-?}"
    sleep 3
}

# Test grid: (dfo, pfo) × (K1, K2) per row.
# K_total = K1 + K2 must equal --k.
# Grid stays compact — 2 splits per (dfo, pfo) for screening.
#
#   dfo  pfo   f   K   K1  K2     test why
#   1    2     3   3   1   2     phase1 short / phase2 longer
#   1    2     3   3   2   1
#   1    3     4   4   2   2     balanced K with small dfo
#   1    3     4   4   1   3
#   2    3     5   5   3   2     known-good baseline (current best at exit=40)
#   2    3     5   5   2   3
#   2    4     6   6   2   4     larger K2 hypothesis (early exit favors big K2)
#   2    4     6   6   3   3
#   3    4     7   7   3   4     larger dfo, balanced split
#   3    4     7   7   4   3
#   3    5     8   8   3   5
#   3    5     8   8   4   4
#   4    5     9   9   4   5     largest fanout combo
#   4    5     9   9   5   4
#   4    6    10   8   3   5     pfo even bigger w/ same K=8 cap
#   4    6    10   8   4   4

run "dfo1_pfo2_K3_K1_1_K2_2" --k 3 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 1 --mesa_phase2_k 2
run "dfo1_pfo2_K3_K1_2_K2_1" --k 3 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 2 --mesa_phase2_k 1
run "dfo1_pfo3_K4_K1_2_K2_2" --k 4 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 2 --mesa_phase2_k 2
run "dfo1_pfo3_K4_K1_1_K2_3" --k 4 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 1 --mesa_phase2_k 3

run "dfo2_pfo3_K5_K1_3_K2_2" --k 5 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 2
run "dfo2_pfo3_K5_K1_2_K2_3" --k 5 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 3
run "dfo2_pfo4_K6_K1_2_K2_4" --k 6 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 4
run "dfo2_pfo4_K6_K1_3_K2_3" --k 6 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 3

run "dfo3_pfo4_K7_K1_3_K2_4" --k 7 --f 7 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 3 --mesa_phase2_k 4
run "dfo3_pfo4_K7_K1_4_K2_3" --k 7 --f 7 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 4 --mesa_phase2_k 3
run "dfo3_pfo5_K8_K1_3_K2_5" --k 8 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 3 --mesa_phase2_k 5
run "dfo3_pfo5_K8_K1_4_K2_4" --k 8 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 4 --mesa_phase2_k 4

run "dfo4_pfo5_K9_K1_4_K2_5" --k 9 --f 9 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 4 --mesa_phase2_k 5
run "dfo4_pfo5_K9_K1_5_K2_4" --k 9 --f 9 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 5 --mesa_phase2_k 4
run "dfo4_pfo6_K8_K1_3_K2_5" --k 8 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 3 --mesa_phase2_k 5
run "dfo4_pfo6_K8_K1_4_K2_4" --k 8 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 4 --mesa_phase2_k 4

echo ""
echo "===== SUMMARY ====="
"$PY" bench/extract_sweep_metrics.py "$OUT" | tee "$OUT/SUMMARY.md"

echo ""
echo "===== Generating timeline plots per config ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    [ -f "$d/run.log" ] || continue
    grep -q "Total Throughput" "$d/run.log" || continue
    echo "  $label"
    "$PY" bench/plot_mesa_timeline.py "$d" --step 50 --warmup 5 2>&1 | tail -1
    "$PY" bench/plot_mesa_breakdown.py "$d" 2>&1 | tail -1
done

echo ""
echo "===== Per-step idle stats (cache-hit steps only) ====="
"$PY" experiments/hybrid_timing_sweep/extract_idle.py "$OUT" 2>&1 | tee "$OUT/IDLE.md"
