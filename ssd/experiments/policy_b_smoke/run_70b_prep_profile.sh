#!/bin/bash
# Fine-grained prep breakdown — find actual bottlenecks.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_PREP=1     # ★ enable fine-grained prep timing

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly"; exit 1
fi
echo "[gpu-set] $CUDA_VISIBLE_DEVICES"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="$PWD/experiments/policy_b_smoke/out_70b_prep"
mkdir -p "$OUT"

# n=3 sequences, output=20 — short for analysis (~60 verify steps).
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 20 --max_model_len 1024 --random --numseqs 3 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

# Best config: K1=8, K2=4, pfo=1
echo "[$(date +%H:%M:%S)] === 70B prep profile (K1=8 K2=4 pfo=1) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
"$PY" -O bench/bench.py $COMMON $QUANT \
    --k 12 --f 3 --mesa --mesa_exit_layer 57 \
    --mesa_phase1_k 8 --mesa_phase2_k 4 \
    --mesa_draft_fan_out 2 --mesa_policy b \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
echo "  rc=$rc TP=${tp:-?}"

echo ""
echo "===== PROFILE_PREP breakdown ====="
"$PY" -c "
import json, glob, statistics as st
files = sorted(glob.glob('$OUT/prep_breakdown_pid*.json'))
print(f'profile files: {len(files)}')
for f in files:
    print(f'--- {f.split(chr(47))[-1]} ---')
    d = json.load(open(f))
    print(f'{\"key\":<40} {\"n\":>5} {\"med (μs)\":>10} {\"mean (μs)\":>10} {\"sum (ms)\":>9}')
    for k in sorted(d.keys()):
        v = d[k]
        n = len(v)
        med = st.median(v) / 1000.0
        mean = sum(v)/n / 1000.0
        s = sum(v) / 1_000_000.0
        print(f'{k:<40} {n:>5} {med:>10.2f} {mean:>10.2f} {s:>9.2f}')
"

