#!/bin/bash
# 70B Policy B + non-uniform Phase 1 fan-out list e2e test.
# Compares uniform vs uniform-via-list (parity) vs non-uniform.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_MESA=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

# K1=8, K2=4, dfo=2, pfo=1. Non-uniform: [3,3,2,2,2,2,2,2,2] sum=20 = uniform.
# Test cases:
# - uniform_implicit: no list (default [2]*9)
# - uniform_via_list: [2,2,2,2,2,2,2,2,2] (parity check)
# - nonuniform_front: [4,4,2,2,2,2,1,1,2] sum=20 (front-loaded)
# - nonuniform_back:  [1,1,2,2,2,2,3,3,4] sum=20 (back-loaded)
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run_case() {
    local tag=$1; local p1list=$2
    local OUT="$PWD/experiments/policy_b_smoke/70b_p1_nonuniform/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (p1_list=$p1list) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    local extra=""
    [ -n "$p1list" ] && extra="--mesa_split_phase1_fan_out_list $p1list"
    "$PY" -O bench/bench.py $COMMON $QUANT \
        --k 12 --f 3 --mesa --mesa_exit_layer 57 \
        --mesa_phase1_k 8 --mesa_phase2_k 4 \
        --mesa_draft_fan_out 2 --mesa_policy b $extra \
        >"$OUT/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|FAIL|AssertionError|NotImplementedError' "$OUT/run.log" | head -2)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 3
}

# Note: K1=8, so list length = 9. dfo=2 → uniform sum = 18.
run_case "uniform_implicit"    ""
run_case "uniform_via_list"    "2,2,2,2,2,2,2,2,2"
run_case "nonuniform_front"    "4,4,2,2,2,2,1,1,0"
run_case "nonuniform_back"     "0,1,1,2,2,2,2,4,4"

echo ""
echo "===== summary ====="
"$PY" -c "
import re
print(f'{\"config\":<22} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"P2_hit\":>7} {\"target_v\":>9} {\"draft_ms\":>9}')
for tag in ['uniform_implicit', 'uniform_via_list', 'nonuniform_front', 'nonuniform_back']:
    d = f'$PWD/experiments/policy_b_smoke/70b_p1_nonuniform/{tag}'
    log = open(f'{d}/run.log').read()
    m = re.search(r'Total Throughput:\s*([\d.]+)', log)
    if not m:
        print(f'{tag:<22} (FAILED)')
        continue
    tp = m.group(1)
    al = re.search(r'Avg Tokens per step \(incl recovery\):\s*([\d.]+)', log).group(1)
    ar = re.search(r'Avg Fraction.*Accepted:\s*([\d.]+)', log).group(1)
    p1h = re.search(r'Avg Phase 1.*Hit Rate:\s*([\d.]+)', log).group(1)
    p2h = re.search(r'Avg Phase 2.*Hit Rate:\s*([\d.]+)', log).group(1)
    tv = re.search(r'Avg target verify time.*:\s*([\d.]+)', log).group(1)
    dms = re.search(r'Avg draft step time.*:\s*([\d.]+)', log).group(1)
    print(f'{tag:<22} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {p2h:>7} {tv:>9} {dms:>9}')
"
