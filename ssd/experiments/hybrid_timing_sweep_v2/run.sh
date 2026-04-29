#!/bin/bash
# Predictor-driven candidates including pfo=dfo (per user request).
# T_p1, T_p2 for pfo=dfo combos extrapolated from prior pfo>dfo sweep:
#   T_p1 ≈ dfo × 1.0 ms/forward (scales with phase1 MQ_LEN)
#   T_p2 ≈ (dfo+pfo) × 0.85 ms/forward (scales with hybrid MQ_LEN)
#   T_glue ≈ 10 ms, proxy_arrival ≈ 35 ms (fanout-independent target side)
# → predicted K1 for pfo=dfo: ≈ (35-10)/T_p1
#     dfo=2 → K1 ≈ 12 (capped to 9 by K_max=10)
#     dfo=3 → K1 ≈ 8
#     dfo=4 → K1 ≈ 6

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
OUT="${1:-$PWD/experiments/hybrid_timing_sweep_v2/results}"
mkdir -p "$OUT"
echo "Output: $OUT"

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run() {
    local tag="$1"; shift
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    if [ -f "$subdir/run.log" ] && grep -q "Total Throughput" "$subdir/run.log"; then
        echo "[skip] $tag (already done)"; return
    fi
    echo "[$(date +%H:%M:%S)] === $tag === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
    SSD_PROFILE_DIR="$subdir" "$PY" -O bench/bench.py $COMMON $QUANT "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?}"; sleep 3
}

# === pfo=dfo (extrapolated K1) — top-2 each ===
run "dfo2_pfo2_K10_K1_9_K2_1" --k 10 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 1
run "dfo2_pfo2_K10_K1_8_K2_2" --k 10 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 2

run "dfo3_pfo3_K10_K1_8_K2_2" --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo3_pfo3_K10_K1_7_K2_3" --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 7 --mesa_phase2_k 3

run "dfo4_pfo4_K8_K1_6_K2_2"  --k 8  --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 6 --mesa_phase2_k 2
run "dfo4_pfo4_K8_K1_7_K2_1"  --k 8  --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 1

# === pfo>dfo (predictor top-2 from v1 sweep data) ===
run "dfo3_pfo4_K10_K1_8_K2_2" --k 10 --f 7 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo3_pfo4_K10_K1_7_K2_3" --k 10 --f 7 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 7 --mesa_phase2_k 3

run "dfo3_pfo5_K9_K1_7_K2_2"  --k 9  --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 7 --mesa_phase2_k 2
run "dfo3_pfo5_K10_K1_8_K2_2" --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2

run "dfo4_pfo5_K8_K1_7_K2_1"  --k 8  --f 9 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 1
run "dfo4_pfo5_K8_K1_6_K2_2"  --k 8  --f 9 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 6 --mesa_phase2_k 2

run "dfo4_pfo6_K8_K1_7_K2_1"  --k 8  --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 1
run "dfo4_pfo6_K8_K1_6_K2_2"  --k 8  --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 6 --mesa_phase2_k 2

echo ""
echo "===== timeline + breakdown plots ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue; [ -f "$d/run.log" ] || continue
    grep -q "Total Throughput" "$d/run.log" || continue
    "$PY" bench/plot_mesa_timeline.py "$d" --step 50 --warmup 5 >/dev/null 2>&1
    "$PY" bench/plot_mesa_breakdown.py "$d" >/dev/null 2>&1
done

echo ""
echo "===== predictor + idle table ====="
"$PY" experiments/hybrid_timing_sweep/predictor.py "$OUT" 2>&1 | tee "$OUT/PREDICTOR.md" | tail -50
echo ""
"$PY" experiments/hybrid_timing_sweep/extract_idle.py "$OUT" 2>&1 | tee "$OUT/IDLE.md" | tail -40
