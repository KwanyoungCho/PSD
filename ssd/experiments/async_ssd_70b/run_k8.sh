#!/bin/bash
# Async SSD baseline (no MESA): k=8, f=2 — fair comparison vs MESA K_max=8.
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
    [ "$n_free" -lt "$N_GPUS_NEEDED" ] && { echo "ERROR: only $n_free GPUs"; exit 1; }
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="$PWD/experiments/async_ssd_70b/results/k8_f2"
mkdir -p "$OUT"

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

echo "[$(date +%H:%M:%S)] === async SSD k=8 f=2 (no MESA) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
SSD_PROFILE_DIR="$OUT" \
    "$PY" -O bench/bench.py $COMMON $QUANT --k 8 --f 2 \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
echo "  rc=$rc TP=${tp:-?}"

echo ""
echo "===== plots ====="
"$PY" bench/plot_mesa_timeline.py "$OUT" --step 50 --warmup 5 >/dev/null 2>&1
"$PY" bench/plot_mesa_breakdown.py "$OUT" >/dev/null 2>&1
ls "$OUT"/*.png 2>/dev/null

echo ""
echo "===== metrics ====="
log="$OUT/run.log"
grep -E "Total Throughput|Avg Tokens per step|Avg Fraction|Avg Cache|Avg Phase|Avg draft step|Avg target verify|Avg Tokens per step on" "$log" | head -10
