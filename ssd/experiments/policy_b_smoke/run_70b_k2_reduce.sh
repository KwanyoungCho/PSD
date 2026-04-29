#!/bin/bash
# K2 reduce hypothesis test: phase 2 seed = "tokens draft tends to mispredict"
# → draft K2 forwards from such seeds yield wrong-context extensions
# → reducing K2 limits wasted extension, freed compute → wider cache (pfo↑)
#
# 3-way:
# - baseline: K1=8, K2=8, pfo=1 (current, k=16, f=3)
# - K2_4:     K1=8, K2=4, pfo=1 (k=12, f=3) — fewer forwards per row
# - K2_4_pfo2: K1=8, K2=4, pfo=2 (k=12, f=4) — fewer forwards × more rows
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
    local tag=$1; local k=$2; local f=$3; local k1=$4; local k2=$5
    local OUT="$PWD/experiments/policy_b_smoke/70b_k2_reduce/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (k=$k, f=$f, K1=$k1, K2=$k2) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    SSD_PROFILE_DIR="$OUT" \
        "$PY" -O bench/bench.py $COMMON $QUANT \
        --k "$k" --f "$f" --mesa --mesa_exit_layer 57 \
        --mesa_phase1_k "$k1" --mesa_phase2_k "$k2" \
        --mesa_draft_fan_out 2 --mesa_policy b \
        >"$OUT/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|FAIL|AssertionError' "$OUT/run.log" | head -2)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 3
}

run_case "baseline_K2_8"  16 3 8 8
run_case "K2_4_pfo1"      12 3 8 4
run_case "K2_4_pfo2"      12 4 8 4

echo ""
echo "===== summary ====="
"$PY" -c "
import re, glob
print(f'{\"config\":<18} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"P2_hit\":>7} {\"target_v\":>9} {\"draft_ms\":>9} {\"rej_on_hit_0\":>13}')
for tag in ['baseline_K2_8', 'K2_4_pfo1', 'K2_4_pfo2']:
    d = f'$PWD/experiments/policy_b_smoke/70b_k2_reduce/{tag}'
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
    print(f'{tag:<18} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {p2h:>7} {tv:>9} {dms:>9} {rej0_val:>13}')

# Also accept_len breakdown from accept_probs (if available)
print()
print('=== accept_suffix breakdown (frequencies) ===')
for tag in ['baseline_K2_8', 'K2_4_pfo1', 'K2_4_pfo2']:
    d = f'$PWD/experiments/policy_b_smoke/70b_k2_reduce/{tag}'
    log = open(f'{d}/run.log').read()
    m = re.search(r'Empirical frequencies of accepted_suffix_lens_on_hit - 1:\s*((?:\s+\d+:\s+\d+\.\d+\s*)+)', log)
    if m:
        print(f'  {tag}:')
        for line in m.group(1).strip().split(chr(10))[:8]:
            print(f'    {line.strip()}')
"
