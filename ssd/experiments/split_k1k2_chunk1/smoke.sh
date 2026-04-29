#!/bin/bash
# Chunk 1 smoke: verify init succeeds with/without SSD_FORCE_SPLIT_K1K2
# (no behavior change yet — just CG capture)

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12397

N_GPUS_NEEDED=3
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
TARGET=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT=/data2/chokwans99/models/Llama-3.2-1B-Instruct
OUT="$PWD/experiments/split_k1k2_chunk1"
mkdir -p "$OUT"

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 3 --b 1 --temp 0 --output_len 32 --max_model_len 2048 --random --numseqs 3 --k 5 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 2"

echo "=== test 1: no env (legacy hybrid path) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
"$PY" -O bench/bench.py $COMMON >"$OUT/no_env.log" 2>&1
rc1=$?
tp1=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/no_env.log" | head -1)
echo "  rc=$rc1 TP=${tp1:-?}"
grep -E "split_k1|split_k2|MESA" "$OUT/no_env.log" | head -5

echo ""
echo "=== test 2: SSD_FORCE_SPLIT_K1K2=1 (init only — no runtime path yet) ==="
sleep 3
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
SSD_FORCE_SPLIT_K1K2=1 "$PY" -O bench/bench.py $COMMON >"$OUT/with_env.log" 2>&1
rc2=$?
tp2=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/with_env.log" | head -1)
echo "  rc=$rc2 TP=${tp2:-?}"
grep -E "split_k1|split_k2|MESA split-K1K2" "$OUT/with_env.log" | head -5
echo ""
err2=$(grep -oP 'Traceback|Error|FAIL' "$OUT/with_env.log" | head -1)
[ -n "$err2" ] && echo "  ERROR: $err2 — last 30 lines:" && tail -30 "$OUT/with_env.log"
