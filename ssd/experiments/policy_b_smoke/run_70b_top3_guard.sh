#!/bin/bash
# 3-way: Policy A vs Policy B vs Policy B + top-3 h-position guard.
# Test if forcing coverage at top-3 reject positions improves hit quality.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_PROFILE_MESA=1
export SSD_FORCE_SPLIT_K1K2=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"
echo "[host-load] $(uptime | awk -F'load average:' '{print $2}')"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run_case() {
    local tag=$1; local policy=$2; local guard=$3
    local OUT="$PWD/experiments/policy_b_smoke/70b_top3_guard/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (policy=$policy, top3_guard=$guard) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    SSD_PROFILE_DIR="$OUT" \
    SSD_MESA_POLICY_B_TOP3_GUARD="$guard" \
        "$PY" -O bench/bench.py $COMMON $QUANT \
        --k 16 --f 3 --mesa --mesa_exit_layer 57 \
        --mesa_phase1_k 8 --mesa_phase2_k 8 \
        --mesa_draft_fan_out 2 --mesa_policy "$policy" \
        >"$OUT/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|FAIL|AssertionError' "$OUT/run.log" | head -2)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 3
}

run_case "policy_a"      "a" "0"
run_case "policy_b"      "b" "0"
run_case "policy_b_top3" "b" "1"

echo ""
echo "===== summary ====="
"$PY" -c "
import re, glob, json, statistics
print(f'{\"config\":<16} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"P2_hit\":>7} {\"target_v\":>9} {\"draft_ms\":>9} {\"rej_on_hit_0\":>13}')
for tag in ['policy_a', 'policy_b', 'policy_b_top3']:
    d = f'$PWD/experiments/policy_b_smoke/70b_top3_guard/{tag}'
    log = open(f'{d}/run.log').read()
    m = re.search(r'Total Throughput:\s*([\d.]+)', log)
    if not m: continue
    tp = m.group(1)
    al = re.search(r'Avg Tokens per step \(incl recovery\):\s*([\d.]+)', log).group(1)
    ar = re.search(r'Avg Fraction.*Accepted:\s*([\d.]+)', log).group(1)
    p1h = re.search(r'Avg Phase 1.*Hit Rate:\s*([\d.]+)', log).group(1)
    p2h = re.search(r'Avg Phase 2.*Hit Rate:\s*([\d.]+)', log).group(1)
    tv = re.search(r'Avg target verify time.*:\s*([\d.]+)', log).group(1)
    dms = re.search(r'Avg draft step time.*:\s*([\d.]+)', log).group(1)
    rej0 = re.search(r'Empirical frequencies of accepted_suffix_lens_on_hit - 1:\s*((?:\s+\d+:\s+\d+\.\d+\s*)+)', log)
    rej0_val = '?'
    if rej0:
        m0 = re.search(r'^\s*0:\s*([\d.]+)', rej0.group(1), re.MULTILINE)
        if m0: rej0_val = m0.group(1)
    print(f'{tag:<16} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {p2h:>7} {tv:>9} {dms:>9} {rej0_val:>13}')
"
