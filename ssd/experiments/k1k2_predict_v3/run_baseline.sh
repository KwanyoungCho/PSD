#!/bin/bash
# Stage 1: K1=2, K2=2 baseline runs per (dfo, pfo) fanout — fast (~1-2 min each).
# These produce timing measurements that the predictor ingests.

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
OUT="${1:-$PWD/experiments/k1k2_predict_v3/baseline}"
mkdir -p "$OUT"
echo "Output: $OUT"

# Tiny baseline: K1=2, K2=2 → K=4. numseqs=8, output_len=48. Should be ~1-2 min/run.
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 48 --max_model_len 2048 --random --numseqs 8"
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
    SSD_PROFILE_DIR="$subdir" \
        "$PY" -O bench/bench.py $COMMON $QUANT "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?}"; sleep 3
}

# 5 fanouts × 1 baseline (K1=2, K2=2)
run "dfo2_pfo3_K4_K1_2_K2_2" --k 4 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 2
run "dfo3_pfo3_K4_K1_2_K2_2" --k 4 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 2 --mesa_phase2_k 2
run "dfo3_pfo5_K4_K1_2_K2_2" --k 4 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 2 --mesa_phase2_k 2
run "dfo4_pfo4_K4_K1_2_K2_2" --k 4 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 2 --mesa_phase2_k 2
run "dfo4_pfo6_K4_K1_2_K2_2" --k 4 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 2 --mesa_phase2_k 2

echo ""
echo "===== predictor ====="
"$PY" experiments/k1k2_predict_v3/predict.py "$OUT" 10 0.5 | tee "$OUT/predictions.md"
