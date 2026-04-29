#!/bin/bash
# Search for first exit_layer where proxy_wait = 0, decrementing from 60.
# Stops as soon as proxy_wait reaches 0.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12298
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
OUT="$PWD/experiments/split_k1k2_70b/exit_fine"
mkdir -p "$OUT"

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

extract_pw() {
    # Returns median proxy_wait in ms, parsed from JSON profile.
    local d="$1"
    "$PY" -c "
import json, glob, statistics
paths = sorted(glob.glob('$d/mesa_profile_draft_*.json'))
if not paths: print('-1'); exit()
rows = json.load(open(paths[-1]))
pw = [r['ms'] for r in rows if r['label']=='proxy_wait']
if not pw: print('-1'); exit()
print(f'{statistics.median(pw):.3f}')
"
}

for L in 60 59 58 57 56 55 54 53 52 51 50; do
    subdir="$OUT/exit_${L}"
    mkdir -p "$subdir"
    if [ -f "$subdir/run.log" ] && grep -q "Total Throughput" "$subdir/run.log"; then
        echo "[skip] exit_$L (already done)"
    else
        echo "[$(date +%H:%M:%S)] === exit_layer=$L ==="
        fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
        SSD_PROFILE_DIR="$subdir" \
            "$PY" -O bench/bench.py $COMMON $QUANT \
            --k 16 --f 3 --mesa --mesa_exit_layer $L \
            --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 8 \
            >"$subdir/run.log" 2>&1
        rc=$?
        tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$subdir/run.log" | head -1)
        echo "  rc=$rc TP=${tp:-?}"
    fi
    pw=$(extract_pw "$subdir")
    echo "  exit_layer=$L: proxy_wait_median=${pw}ms"
    # Check if proxy_wait <= 0.05 ms (effectively zero — measurement noise tolerance).
    is_zero=$("$PY" -c "print(1 if float('$pw') >= 0 and float('$pw') <= 0.05 else 0)")
    if [ "$is_zero" = "1" ]; then
        echo ""
        echo "★★★ FOUND: exit_layer=$L → proxy_wait=${pw}ms (≈ 0) ★★★"
        break
    fi
    sleep 3
done

echo ""
echo "===== summary so far ====="
"$PY" -c "
import json, glob, re, os
print(f'{\"exit\":>5} {\"TPS\":>7} {\"avg_len\":>8} {\"accept\":>8} {\"proxy_wait_med\":>15} {\"proxy_arr\":>10} {\"phase1_end\":>11}')
for L in range(60, 50, -1):
    d = f'$OUT/exit_{L}'
    if not os.path.exists(f'{d}/run.log'): continue
    log = open(f'{d}/run.log').read()
    try:
        tp = re.search(r'Total Throughput:\s*([\d.]+)', log).group(1)
        al = re.search(r'Avg Tokens per step \(incl recovery\):\s*([\d.]+)', log).group(1)
        ar = re.search(r'Avg Fraction.*Accepted:\s*([\d.]+)', log).group(1)
    except: continue
    paths = sorted(glob.glob(f'{d}/mesa_profile_draft_*.json'))
    if not paths: continue
    rows = json.load(open(paths[-1]))
    pw_evts = sorted(r['ms'] for r in rows if r['label']=='proxy_wait')
    pw_med = pw_evts[len(pw_evts)//2] if pw_evts else 0
    glue_evts = [r['ms'] for r in rows if r['label']=='glue']
    p1_evts = [r['ms'] for r in rows if r['label']=='phase1_replay']
    n_step = max(1, sum(1 for r in rows if r['label']=='glue'))
    phase1_end = sum(glue_evts)/n_step + sum(p1_evts)/n_step
    paths_t = sorted(glob.glob(f'{d}/mesa_profile_target_rank0_*.json'))
    rows_t = json.load(open(paths_t[-1]))
    n_t = max(1, sum(1 for r in rows_t if r['label']=='target_postprocess'))
    gp = sum(r['ms'] for r in rows_t if r['label']=='graph_pre') / n_t
    pcs = sum(r['ms'] for r in rows_t if r['label']=='proxy_compute_send') / n_t
    proxy_arr = gp + pcs
    print(f'{L:>5} {tp:>7} {al:>8} {ar:>8} {pw_med:>15.3f} {proxy_arr:>10.2f} {phase1_end:>11.2f}')
" | tee "$OUT/SUMMARY.txt"
