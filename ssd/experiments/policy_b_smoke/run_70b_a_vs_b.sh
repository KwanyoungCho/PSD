#!/bin/bash
# 70B Policy A vs Policy B head-to-head (split-K1/K2, uniform).
# Same seed → same prompts. temp=0 → deterministic sampling.
# Compares: TPS, accept rate, phase1/2_prep/replay/build, glue.
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

# Same prompts + same sampling (seed=42), temp=0.6 stochastic, n=10
# (per memory: stochastic is the meaningful regime for MESA A/B; --seed
# fixes both --random prompts and torch sampling trajectory)
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run_case() {
    local tag=$1; local policy=$2
    local OUT="$PWD/experiments/policy_b_smoke/70b_a_vs_b_t06/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (policy=$policy, seed=42, temp=0, n=10) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    SSD_PROFILE_DIR="$OUT" \
        "$PY" -O bench/bench.py $COMMON $QUANT \
        --k 16 --f 3 --mesa --mesa_exit_layer 57 \
        --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 8 \
        --mesa_policy "$policy" \
        >"$OUT/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|FAIL|Aborted|AssertionError' "$OUT/run.log" | head -2)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 3
}

run_case "policy_a" "a"
run_case "policy_b" "b"

echo ""
echo "===== summary ====="
"$PY" -c "
import json, glob, re, statistics, os
LABELS_DRAFT = ['glue', 'phase1_build', 'phase1_prep', 'phase1_replay', 'phase2_build', 'phase2_prep', 'phase2_replay']
print(f'{\"config\":<10} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"P2_hit\":>7} {\"target_v\":>9} {\"draft_ms\":>9}')
for tag in ['policy_a', 'policy_b']:
    d = f'$PWD/experiments/policy_b_smoke/70b_a_vs_b_t06/{tag}'
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
    print(f'{tag:<10} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {p2h:>7} {tv:>9} {dms:>9}')
print()
print('=== draft per-event medians (warmup 5) ===')
print(f'{\"event\":<22} {\"polA\":>10} {\"polB\":>10} {\"ratio\":>7}')
for label in LABELS_DRAFT:
    d_a = sorted(glob.glob('$PWD/experiments/policy_b_smoke/70b_a_vs_b_t06/policy_a/mesa_profile_draft_*.json'))[-1]
    d_b = sorted(glob.glob('$PWD/experiments/policy_b_smoke/70b_a_vs_b_t06/policy_b/mesa_profile_draft_*.json'))[-1]
    rows_a = json.load(open(d_a))
    rows_b = json.load(open(d_b))
    e_a = [r['ms'] for r in rows_a if r['label']==label][5:]
    e_b = [r['ms'] for r in rows_b if r['label']==label][5:]
    if not e_a or not e_b: continue
    m_a = statistics.median(e_a); m_b = statistics.median(e_b)
    print(f'{label:<22} {m_a:>10.3f} {m_b:>10.3f} {m_b/m_a:>6.2f}x')
"
