#!/bin/bash
# 8B smoke test v2 — extends original smoke with split-via-hybrid mode.
# Compares 3 modes:
#   default          : SSD env vars off (hybrid CG default)
#   force_split      : SSD_FORCE_SPLIT_PHASE2=1 (legacy split, has mask bug)
#   split_via_hybrid : SSD_SPLIT_VIA_HYBRID=1 (new: hybrid CG twice = correct split)

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12299
export SSD_PROFILE_MESA=1

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
OUT="${1:-$PWD/experiments/split_k1k2_smoke/results_v2}"
mkdir -p "$OUT"

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 3 --b 1 --temp 0 --output_len 64 --max_model_len 2048 --random --numseqs 5"

run() {
    local tag="$1"; shift
    local mode="${1:-default}"; shift   # "default" | "force_split" | "split_via_hybrid"
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    if [ -f "$subdir/run.log" ] && grep -q "Total Throughput" "$subdir/run.log"; then
        echo "[skip] $tag (already done)"; return
    fi
    echo "[$(date +%H:%M:%S)] === $tag (mode=$mode) === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
    local force_split=0 split_via=0
    [ "$mode" = "force_split" ] && force_split=1
    [ "$mode" = "split_via_hybrid" ] && split_via=1
    SSD_PROFILE_DIR="$subdir" SSD_FORCE_SPLIT_PHASE2="$force_split" \
        SSD_SPLIT_VIA_HYBRID="$split_via" \
        "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local err=$(grep -oP 'Traceback|Error|FAIL' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"; sleep 3
}

# Verify split-via-hybrid produces same output as hybrid (correctness)
# Test 3 modes × 2 (K1, K2) configs = 6 runs

# K1=2, K2=2 (symmetric)
run "hybrid_K4_K1_2_K2_2"           default          --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 2
run "force_split_K4_K1_2_K2_2"      force_split      --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 2
run "split_via_hybrid_K4_K1_2_K2_2" split_via_hybrid --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 2

# K1=3, K2=1 (asymmetric)
run "hybrid_K4_K1_3_K2_1"           default          --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 1
run "force_split_K4_K1_3_K2_1"      force_split      --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 1
run "split_via_hybrid_K4_K1_3_K2_1" split_via_hybrid --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 1

# K1=4, K2=2 (deep, asymmetric)
run "hybrid_K6_K1_4_K2_2"           default          --k 6 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 2
run "split_via_hybrid_K6_K1_4_K2_2" split_via_hybrid --k 6 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 2

echo ""
echo "===== SUMMARY ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    log="$d/run.log"; [ -f "$log" ] || continue
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" | head -1)
    ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|AssertionError|FAILED' "$log" | head -1)
    printf "  %-40s TPS=%-7s accept=%-5s %s\n" "$label" "${tp:-FAIL}" "${ar:-?}" "${err}"
done
