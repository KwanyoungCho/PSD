#!/bin/bash
# Same 3-way comparison but with --temp 0 (deterministic sampling).
# If uniform == uniform_via_list under temp=0, prior 0.31/0.39 split was variance.
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

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

# temp 0 + larger numseqs to reduce per-prompt variance
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0 --output_len 64 --max_model_len 2048 --random --numseqs 20"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run_case() {
    local tag=$1; local p1list=$2
    local OUT="$PWD/experiments/split_k1k2_70b/tp4_3way_temp0/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (p1=$p1list, temp=0, n=20) ==="
    fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
    local extra=""
    [ -n "$p1list" ] && extra="--mesa_split_phase1_fan_out_list $p1list"
    SSD_PROFILE_DIR="$OUT" \
        "$PY" -O bench/bench.py $COMMON $QUANT \
        --k 16 --f 3 --mesa --mesa_exit_layer 57 \
        --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 8 $extra \
        >"$OUT/run.log" 2>&1
    rc=$?
    tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
    err=$(grep -oP 'Traceback|RuntimeError|Aborted' "$OUT/run.log" | head -1)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    sleep 3
}

run_case "uniform"          ""
run_case "uniform_via_list" "2,2,2,2,2,2,2,2,2"
run_case "nonuniform"       "5,3,1,1,0,0,0,0,5"

echo ""
echo "===== summary (temp=0, n=20) ====="
"$PY" -c "
import json, glob, re, statistics, os
print(f'{\"config\":<20} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"target_v\":>9} {\"draft_ms\":>10} {\"phase1_md\":>10}')
for tag in ['uniform', 'uniform_via_list', 'nonuniform']:
    d = f'$PWD/experiments/split_k1k2_70b/tp4_3way_temp0/{tag}'
    if not os.path.exists(f'{d}/run.log'): continue
    log = open(f'{d}/run.log').read()
    m = re.search(r'Total Throughput:\s*([\d.]+)', log)
    if not m: continue
    tp = m.group(1)
    al = re.search(r'Avg Tokens per step \(incl recovery\):\s*([\d.]+)', log).group(1)
    ar = re.search(r'Avg Fraction.*Accepted:\s*([\d.]+)', log).group(1)
    p1h = re.search(r'Avg Phase 1.*Hit Rate:\s*([\d.]+)', log).group(1)
    tv = re.search(r'Avg target verify time.*:\s*([\d.]+)', log).group(1)
    dms = re.search(r'Avg draft step time.*:\s*([\d.]+)', log).group(1)
    paths = sorted(glob.glob(f'{d}/mesa_profile_draft_*.json'))
    p1m = 0
    if paths:
        rows = json.load(open(paths[-1]))
        p1_evts = [r['ms'] for r in rows if r['label']=='phase1_replay']
        p1m = statistics.median(p1_evts) if p1_evts else 0
    print(f'{tag:<20} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {tv:>9} {dms:>10} {p1m:>10.3f}')"
