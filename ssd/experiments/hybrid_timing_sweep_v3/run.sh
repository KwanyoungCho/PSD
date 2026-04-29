#!/bin/bash
# v3: dfo ∈ {1, 2}, pfo ≥ dfo. Per user feedback — small fanout means cheap
# forwards, so K1 can go higher (10-11). K2 fills remaining target budget.
#
# Predictor calculation per fanout combo (using v1 + v2 data):
#   dfo=1, T_p1≈1.85, T_glue≈9, proxy_arrival≈27.5
#     → K1_real = (27.5-9)/1.85 = 10.0
#   dfo=2, T_p1≈3.3, T_glue≈14, proxy_arrival≈45
#     → K1_real = (45-14)/3.3 = 9.4
#   K2 = (target_total - T_glue - K1*T_p1 - phase2_build) / T_p2

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
OUT="${1:-$PWD/experiments/hybrid_timing_sweep_v3/results}"
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

# === dfo=1, pfo ∈ {2, 3, 4} (pfo=1 skipped — too narrow tree) ===
# Predictor: K1≈10, K2 fills phase 2 budget per pfo
run "dfo1_pfo2_K13_K1_10_K2_3" --k 13 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 10 --mesa_phase2_k 3
run "dfo1_pfo2_K12_K1_11_K2_1" --k 12 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 11 --mesa_phase2_k 1

run "dfo1_pfo3_K13_K1_10_K2_3" --k 13 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 10 --mesa_phase2_k 3
run "dfo1_pfo3_K13_K1_11_K2_2" --k 13 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 11 --mesa_phase2_k 2

run "dfo1_pfo4_K12_K1_10_K2_2" --k 12 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 10 --mesa_phase2_k 2
run "dfo1_pfo4_K13_K1_11_K2_2" --k 13 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 11 --mesa_phase2_k 2

# === dfo=2, pfo ∈ {2, 3, 4} ===
# Predictor: K1≈9, K2 fills phase 2 budget per pfo
run "dfo2_pfo2_K12_K1_9_K2_3"  --k 12 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 3
run "dfo2_pfo2_K12_K1_10_K2_2" --k 12 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 10 --mesa_phase2_k 2

run "dfo2_pfo3_K13_K1_9_K2_4"  --k 13 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 4
run "dfo2_pfo3_K13_K1_10_K2_3" --k 13 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 10 --mesa_phase2_k 3

run "dfo2_pfo4_K12_K1_9_K2_3"  --k 12 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 3
run "dfo2_pfo4_K12_K1_10_K2_2" --k 12 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 10 --mesa_phase2_k 2

echo ""
echo "===== timeline + breakdown plots ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue; [ -f "$d/run.log" ] || continue
    grep -q "Total Throughput" "$d/run.log" || continue
    "$PY" bench/plot_mesa_timeline.py "$d" --step 50 --warmup 5 >/dev/null 2>&1
    "$PY" bench/plot_mesa_breakdown.py "$d" >/dev/null 2>&1
done

echo ""
echo "===== predictor self-validation ====="
"$PY" experiments/hybrid_timing_sweep/predictor.py "$OUT" 2>&1 | tee "$OUT/PREDICTOR.md" | tail -60
echo ""
"$PY" experiments/hybrid_timing_sweep/extract_idle.py "$OUT" 2>&1 | tee "$OUT/IDLE.md" | tail -30
