#!/bin/bash
# Test: skip _plan_event.synchronize() in run_fi_tree_decode_cudagraph.
# Hypothesis: 2.6 ms/call sync wait is unnecessary if plan() is CPU-side.
# Compare with-sync vs no-sync at same seed.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_FORCE_SPLIT_K1K2=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run_case() {
    local tag=$1; local skip_sync=$2
    local OUT="$PWD/experiments/policy_b_smoke/70b_skip_sync/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (skip_sync=$skip_sync) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    SSD_SKIP_PLAN_SYNC="$skip_sync" \
        "$PY" -O bench/bench.py $COMMON $QUANT \
        --k 12 --f 3 --mesa --mesa_exit_layer 57 \
        --mesa_phase1_k 8 --mesa_phase2_k 4 \
        --mesa_draft_fan_out 2 --mesa_policy b \
        >"$OUT/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|FAIL|AssertionError|CUDA error' "$OUT/run.log" | head -2)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 3
}

run_case "with_sync"  "0"
run_case "skip_sync"  "1"

echo ""
echo "===== summary ====="
"$PY" -c "
import re
print(f'{\"config\":<14} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"P2_hit\":>7} {\"target_v\":>9} {\"draft_ms\":>9}')
for tag in ['with_sync', 'skip_sync']:
    d = f'$PWD/experiments/policy_b_smoke/70b_skip_sync/{tag}'
    log = open(f'{d}/run.log').read()
    m = re.search(r'Total Throughput:\s*([\d.]+)', log)
    if not m:
        print(f'{tag:<14} (FAILED)')
        continue
    tp = m.group(1)
    al = re.search(r'Avg Tokens per step \(incl recovery\):\s*([\d.]+)', log).group(1)
    ar = re.search(r'Avg Fraction.*Accepted:\s*([\d.]+)', log).group(1)
    p1h = re.search(r'Avg Phase 1.*Hit Rate:\s*([\d.]+)', log).group(1)
    p2h = re.search(r'Avg Phase 2.*Hit Rate:\s*([\d.]+)', log).group(1)
    tv = re.search(r'Avg target verify time.*:\s*([\d.]+)', log).group(1)
    dms = re.search(r'Avg draft step time.*:\s*([\d.]+)', log).group(1)
    print(f'{tag:<14} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {p2h:>7} {tv:>9} {dms:>9}')
"
