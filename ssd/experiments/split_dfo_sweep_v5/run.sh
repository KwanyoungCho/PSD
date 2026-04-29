#!/bin/bash
# Split mode dfo sweep at fixed K1=7, K2=7, pfo=1.
# Compare draft_fan_out ∈ {1, 2, 3, 4} on 70B both-AWQ.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export SSD_PROFILE_MESA=1
export SSD_FORCE_SPLIT_PHASE2=1   # ★ split (NEW) mode

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
OUT="${1:-$PWD/experiments/split_dfo_sweep_v5/results}"
mkdir -p "$OUT"
echo "Output: $OUT"
echo "Mode: split (NEW), K1=7, K2=7, pfo=1, exit_layer=40"

# Long runs for clean TPS — numseqs=10, output_len=64. K=14 (K1+K2).
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
    SSD_PROFILE_DIR="$subdir" \
        "$PY" -O bench/bench.py $COMMON $QUANT "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local err=$(grep -oP 'Traceback|Error|FAIL' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"; sleep 3
}

# K1=7, K2=7, K=14, pfo=1, dfo ∈ {1,2,3,4}
# --f = dfo + pfo
run "dfo1_pfo1_K1_7_K2_7" --k 14 --f 2 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 1 --mesa_phase1_k 7 --mesa_phase2_k 7
run "dfo2_pfo1_K1_7_K2_7" --k 14 --f 3 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 7 --mesa_phase2_k 7
run "dfo3_pfo1_K1_7_K2_7" --k 14 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 7 --mesa_phase2_k 7
run "dfo4_pfo1_K1_7_K2_7" --k 14 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 7

echo ""
echo "===== plots ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue; [ -f "$d/run.log" ] || continue
    grep -q "Total Throughput" "$d/run.log" || continue
    "$PY" bench/plot_mesa_timeline.py "$d" --step 50 --warmup 5 >/dev/null 2>&1
    "$PY" bench/plot_mesa_breakdown.py "$d" >/dev/null 2>&1
done

echo ""
echo "===== summary ====="
{
printf "%-30s %-10s %-8s %-8s %-8s %-10s %-10s\n" "config" "TPS" "accept" "P1" "P2" "draft_ms" "verify_ms"
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
    printf "%-30s %-10s %-8s %-8s %-8s %-10s %-10s\n" \
        "$label" "${tp:-FAIL}" "${ar:-?}" "${p1:-?}" "${p2:-?}" "${dms:-?}" "${vms:-?}"
done
} | tee "$OUT/SUMMARY.txt"
