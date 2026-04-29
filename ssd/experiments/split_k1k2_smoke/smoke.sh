#!/bin/bash
# 8B smoke test for split + K1/K2 path.
# Goal: verify the path runs without error (was untested for K1≠K2 before my fix)
# AND check output quality matches hybrid.
#
# Compare 4 modes at the same args:
#   1. Pure legacy split  (no K1/K2)
#   2. Hybrid K1=K2=2 (known working since Step 7 parity)
#   3. Split K1=K2=2  (my fix not strictly needed here but good control)
#   4. Hybrid K1=3, K2=1 (known working)
#   5. Split K1=3, K2=1 (NEW — first time exercised with metadata_ints fix)
#   6. Split K1=4, K2=1 (NEW — even more asymmetric)

set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12299
export SSD_PROFILE_MESA=1

N_GPUS_NEEDED=3   # 8B target TP=2 + 1 draft
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
OUT="${1:-$PWD/experiments/split_k1k2_smoke/results}"
mkdir -p "$OUT"

# Tiny run: 5 prompts × output_len=64 → ~80 spec steps (plenty for sanity)
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 3 --b 1 --temp 0 --output_len 64 --max_model_len 2048 --random --numseqs 5"

run() {
    local tag="$1"; shift
    local force_split="${1:-0}"; shift  # first arg: 0 or 1 for SSD_FORCE_SPLIT_PHASE2
    local subdir="$OUT/$tag"
    mkdir -p "$subdir"
    if [ -f "$subdir/run.log" ] && grep -q "Total Throughput" "$subdir/run.log"; then
        echo "[skip] $tag (already done)"; return
    fi
    echo "[$(date +%H:%M:%S)] === $tag (FORCE_SPLIT=$force_split) === $*"
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null || true; sleep 2
    SSD_PROFILE_DIR="$subdir" SSD_FORCE_SPLIT_PHASE2="$force_split" \
        "$PY" -O bench/bench.py $COMMON "$@" >"$subdir/run.log" 2>&1
    local rc=$?
    local tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    local err=$(grep -oP 'Traceback|Error|FAIL' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"; sleep 3
}

# 1. Pure legacy split (K1/K2 미설정)
run "legacy_K4"            0 --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2

# 2. Hybrid K1=K2=2
run "hybrid_K4_K1_2_K2_2"  0 --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 2

# 3. Split K1=K2=2 (control, old code might have worked)
run "split_K4_K1_2_K2_2"   1 --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 2

# 4. Hybrid K1=3, K2=1
run "hybrid_K4_K1_3_K2_1"  0 --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 1

# 5. Split K1=3, K2=1 — NEW path (was broken before fix)
run "split_K4_K1_3_K2_1"   1 --k 4 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 1

# 6. Split K1=4, K2=1 (asymmetric, bigger gap)
run "split_K5_K1_4_K2_1"   1 --k 5 --f 4 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 1

echo ""
echo "===== SUMMARY ====="
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue
    label=$(basename "$d")
    log="$d/run.log"
    [ -f "$log" ] || continue
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" | head -1)
    ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|AssertionError|FAILED' "$log" | head -1)
    printf "  %-25s TPS=%-7s accept=%-5s %s\n" "$label" "${tp:-FAIL}" "${ar:-?}" "${err}"
done
