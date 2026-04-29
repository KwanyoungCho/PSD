#!/bin/bash
# K1 nearby-search: for each fanout, run K1 ∈ small range (3-4 values), K2 = K_max - K1.
# Short runs (numseqs=5, output_len=32) — ~40-50s/run. Pick best K1 per fanout.
# Existing v2 results re-used; we only fill missing K1 values.

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12299
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
OUT="${1:-$PWD/experiments/k1_sweep_v4/results}"
mkdir -p "$OUT"
echo "Output: $OUT"

# SHORT runs — numseqs=5, output_len=32 → ~40-50s/run
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 32 --max_model_len 2048 --random --numseqs 5"
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
    echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"; sleep 2
}

# K_max=10 (K2 = 10 - K1)
# dfo=2, pfo=3
run "dfo2_pfo3_K1_6_K2_4"  --k 10 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 6 --mesa_phase2_k 4
run "dfo2_pfo3_K1_7_K2_3"  --k 10 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 7 --mesa_phase2_k 3
run "dfo2_pfo3_K1_8_K2_2"  --k 10 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo2_pfo3_K1_9_K2_1"  --k 10 --f 5 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 9 --mesa_phase2_k 1

# dfo=3, pfo=3
run "dfo3_pfo3_K1_6_K2_4"  --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 6 --mesa_phase2_k 4
run "dfo3_pfo3_K1_7_K2_3"  --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 7 --mesa_phase2_k 3
run "dfo3_pfo3_K1_8_K2_2"  --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo3_pfo3_K1_9_K2_1"  --k 10 --f 6 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 9 --mesa_phase2_k 1

# dfo=3, pfo=5
run "dfo3_pfo5_K1_6_K2_4"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 6 --mesa_phase2_k 4
run "dfo3_pfo5_K1_7_K2_3"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 7 --mesa_phase2_k 3
run "dfo3_pfo5_K1_8_K2_2"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 8 --mesa_phase2_k 2
run "dfo3_pfo5_K1_9_K2_1"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 3 --mesa_phase1_k 9 --mesa_phase2_k 1

# dfo=4, pfo=4
run "dfo4_pfo4_K1_5_K2_5"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 5 --mesa_phase2_k 5
run "dfo4_pfo4_K1_6_K2_4"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 6 --mesa_phase2_k 4
run "dfo4_pfo4_K1_7_K2_3"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 3
run "dfo4_pfo4_K1_8_K2_2"  --k 10 --f 8 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 8 --mesa_phase2_k 2

# dfo=4, pfo=6
run "dfo4_pfo6_K1_5_K2_5"  --k 10 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 5 --mesa_phase2_k 5
run "dfo4_pfo6_K1_6_K2_4"  --k 10 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 6 --mesa_phase2_k 4
run "dfo4_pfo6_K1_7_K2_3"  --k 10 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 7 --mesa_phase2_k 3
run "dfo4_pfo6_K1_8_K2_2"  --k 10 --f 10 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 4 --mesa_phase1_k 8 --mesa_phase2_k 2

echo ""
echo "===== summary ====="
{
printf "%-30s %-10s %-8s %-8s %-8s %-10s\n" "config" "TPS" "accept" "P1" "P2" "draft_ms"
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    log="$d/run.log"; [ -f "$log" ] || continue
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" | head -1)
    ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$log" | head -1)
    p1=$(grep -oP 'Avg Phase 1 \(draft\) Hit Rate:\s*\K[\d.]+' "$log" | head -1)
    p2=$(grep -oP 'Avg Phase 2 \(proxy\) Hit Rate:\s*\K[\d.]+' "$log" | head -1)
    dms=$(grep -oP 'Avg draft step time \(ms\):\s*\K[\d.]+' "$log" | head -1)
    printf "%-30s %-10s %-8s %-8s %-8s %-10s\n" \
        "$label" "${tp:-FAIL}" "${ar:-?}" "${p1:-?}" "${p2:-?}" "${dms:-?}"
done
} | tee "$OUT/SUMMARY.txt"
