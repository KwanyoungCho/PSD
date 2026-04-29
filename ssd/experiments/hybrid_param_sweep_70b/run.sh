#!/bin/bash
# Phase 2 hybrid parameter sweep — both AWQ on layerskip-llama2-70B + TinyLlama-1.1B AWQ.
#
# Hypothesis (user-supplied):
#   K1 ≈ early-exit position (later exit → larger K1, more time before proxy arrives).
#   K2 chosen to minimize idle (proxy_wait + draft_recv_cmd) — i.e. K2 should
#   roughly bridge the gap between Phase 1 finish and proxy arrival.
#
# Sweep grid: 3 exit layers × 4 (K1,K2) configs = 12 runs.
#   Exit selection (out of L=80 for layerskip-llama2-70B):
#     40 (= 1/2 L)  early exit  — small K1 hypothesis (more proxy reliance)
#     47 (= 7/12 L) mid exit    — medium K1
#     53 (= 2/3 L)  late exit   — larger K1 (more pre-proxy budget)

set -u
cd "$(dirname "$0")/../.."   # ssd repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
export SSD_PROFILE_MESA=1

# Auto-pick free GPUs (mem<1GiB AND util<5%)
N_GPUS_NEEDED=5
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v N=$N_GPUS_NEEDED '
            { gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }
        ' | head -n $N_GPUS_NEEDED | paste -sd,
    )
    n_free=$(echo "$free_gpus" | tr ',' '\n' | grep -c .)
    if [ "$n_free" -lt "$N_GPUS_NEEDED" ]; then
        echo "ERROR: only $n_free free GPUs found (need $N_GPUS_NEEDED). Aborting." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] using free GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "[gpu-pick] CUDA_VISIBLE_DEVICES override: $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="${1:-$PWD/experiments/hybrid_param_sweep_70b/results}"
mkdir -p "$OUT"
echo "Output: $OUT"

# 200 prompts × output_len=256, both AWQ
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 256 --max_model_len 2048 --all --numseqs 50"
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
    local ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$subdir/run.log" | head -1)
    echo "  rc=$rc TP=${tp:-?} accept=${ar:-?}"
    sleep 3
}

# ============================================================================
# exit_layer = 40  (early exit, 1/2 L) — hypothesis: smaller K1
# ============================================================================
run "ex40_K5_K1_3_K2_2" --k 5 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 2
run "ex40_K5_K1_2_K2_3" --k 5 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 3
run "ex40_K6_K1_3_K2_3" --k 6 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 3
run "ex40_K6_K1_2_K2_4" --k 6 --f 4 --mesa --mesa_exit_layer 40 --mesa_draft_fan_out 2 --mesa_phase1_k 2 --mesa_phase2_k 4

# ============================================================================
# exit_layer = 47  (mid exit, 7/12 L) — hypothesis: medium K1
# ============================================================================
run "ex47_K5_K1_3_K2_2" --k 5 --f 4 --mesa --mesa_exit_layer 47 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 2
run "ex47_K6_K1_3_K2_3" --k 6 --f 4 --mesa --mesa_exit_layer 47 --mesa_draft_fan_out 2 --mesa_phase1_k 3 --mesa_phase2_k 3
run "ex47_K6_K1_4_K2_2" --k 6 --f 4 --mesa --mesa_exit_layer 47 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 2
run "ex47_K7_K1_4_K2_3" --k 7 --f 4 --mesa --mesa_exit_layer 47 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 3

# ============================================================================
# exit_layer = 53  (late exit, 2/3 L) — hypothesis: larger K1
# ============================================================================
run "ex53_K6_K1_4_K2_2" --k 6 --f 4 --mesa --mesa_exit_layer 53 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 2
run "ex53_K7_K1_4_K2_3" --k 7 --f 4 --mesa --mesa_exit_layer 53 --mesa_draft_fan_out 2 --mesa_phase1_k 4 --mesa_phase2_k 3
run "ex53_K7_K1_5_K2_2" --k 7 --f 4 --mesa --mesa_exit_layer 53 --mesa_draft_fan_out 2 --mesa_phase1_k 5 --mesa_phase2_k 2
run "ex53_K8_K1_5_K2_3" --k 8 --f 4 --mesa --mesa_exit_layer 53 --mesa_draft_fan_out 2 --mesa_phase1_k 5 --mesa_phase2_k 3

echo ""
echo "===== SUMMARY ====="
"$PY" bench/extract_sweep_metrics.py "$OUT" | tee "$OUT/SUMMARY.md"

echo ""
echo "===== Best by exit_layer ====="
for ex in 40 47 53; do
    echo "exit_layer=$ex:"
    for d in "$OUT"/ex${ex}_*/; do
        log="$d/run.log"; [ -f "$log" ] || continue
        tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$log" | head -1)
        ar=$(grep -oP 'Avg Fraction of Speculated Tokens Accepted:\s*\K[\d.]+' "$log" | head -1)
        pw=$(grep -oP 'Avg.*proxy.*wait.*ms.*:\s*\K[\d.]+' "$log" | head -1)
        ds=$(grep -oP 'Avg draft step time \(ms\):\s*\K[\d.]+' "$log" | head -1)
        printf "  %-22s TPS=%-7s accept=%-5s draft_ms=%-7s\n" "$(basename $d)" "${tp:-?}" "${ar:-?}" "${ds:-?}"
    done
done
