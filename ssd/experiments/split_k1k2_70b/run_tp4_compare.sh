#!/bin/bash
# TP=4: uniform vs non-uniform [5,3,1,1,0,0,0,0,5] at exit_layer=57.
# Same GPU set, fair comparison.
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
    free_gpus=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ gsub(/ /,""); if ($2 < 1024 && $3 < 5) print $1 }' \
        | head -5 | paste -sd,
    )
    [ "$(echo $free_gpus | tr ',' '\n' | wc -l)" -lt 5 ] && { echo "ERROR: need 5 GPUs"; exit 1; }
    export CUDA_VISIBLE_DEVICES="$free_gpus"
    echo "[gpu-pick] $CUDA_VISIBLE_DEVICES"
fi

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

run_case() {
    local tag=$1; local p1list=$2
    local OUT="$PWD/experiments/split_k1k2_70b/tp4_compare/$tag"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] === $tag (TP=4, exit=57, p1=$p1list) ==="
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
    err=$(grep -oP 'Traceback|RuntimeError|Aborted|FAIL' "$OUT/run.log" | head -1)
    echo "  $tag: rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
    "$PY" bench/plot_mesa_timeline.py "$OUT" --step 50 --warmup 5 >/dev/null 2>&1
    "$PY" bench/plot_mesa_breakdown.py "$OUT" >/dev/null 2>&1
    sleep 3
}

run_case "uniform"     ""
run_case "nonuniform"  "5,3,1,1,0,0,0,0,5"

echo ""
echo "===== summary ====="
"$PY" -c "
import json, glob, re, statistics, os
print(f'{\"config\":<14} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"P1_hit\":>7} {\"graph_pre\":>11} {\"phase1_med\":>11} {\"phase2_med\":>11} {\"draft_ms\":>10}')
for tag in ['uniform','nonuniform']:
    d = f'$PWD/experiments/split_k1k2_70b/tp4_compare/{tag}'
    if not os.path.exists(f'{d}/run.log'): continue
    log = open(f'{d}/run.log').read()
    m = re.search(r'Total Throughput:\s*([\d.]+)', log)
    if not m: continue
    tp = m.group(1)
    al = re.search(r'Avg Tokens per step \(incl recovery\):\s*([\d.]+)', log).group(1)
    ar = re.search(r'Avg Fraction.*Accepted:\s*([\d.]+)', log).group(1)
    p1h = re.search(r'Avg Phase 1.*Hit Rate:\s*([\d.]+)', log).group(1)
    dms = re.search(r'Avg draft step time.*:\s*([\d.]+)', log).group(1)
    paths = sorted(glob.glob(f'{d}/mesa_profile_draft_*.json'))
    rows = json.load(open(paths[-1]))
    p1_evts = [r['ms'] for r in rows if r['label']=='phase1_replay']
    p2_evts = [r['ms'] for r in rows if r['label']=='phase2_replay']
    p1m = statistics.median(p1_evts) if p1_evts else 0
    p2m = statistics.median(p2_evts) if p2_evts else 0
    paths_t = sorted(glob.glob(f'{d}/mesa_profile_target_rank0_*.json'))
    rows_t = json.load(open(paths_t[-1]))
    gp_evts = [r['ms'] for r in rows_t if r['label']=='graph_pre']
    gpm = statistics.median(gp_evts) if gp_evts else 0
    print(f'{tag:<14} {tp:>7} {al:>8} {ar:>8} {p1h:>7} {gpm:>11.2f} {p1m:>11.3f} {p2m:>11.3f} {dms:>10}')"
