#!/bin/bash
# Test K_loop = layout.K fix.
# Baseline (before fix) was K2_4_pfo1 = 45.85 TPS in earlier 3-way.
# Same config: K1=8, K2=4, pfo=1, exit=57.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_DIST_PORT=12399
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_PREP=1
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

OUT="$PWD/experiments/policy_b_smoke/out_70b_kloop_fix"
mkdir -p "$OUT"
rm -f /tmp/prep_breakdown_pid*.json

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10 --seed 42"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

echo "[$(date +%H:%M:%S)] === K2_4_pfo1 with K_loop=layout.K fix ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
SSD_PROFILE_DIR="$OUT" \
    "$PY" -O bench/bench.py $COMMON $QUANT \
    --k 12 --f 3 --mesa --mesa_exit_layer 57 \
    --mesa_phase1_k 8 --mesa_phase2_k 4 \
    --mesa_draft_fan_out 2 --mesa_policy b \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
err=$(grep -oP 'Traceback|RuntimeError|FAIL|AssertionError' "$OUT/run.log" | head -2)
echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"

# move pid file to OUT
mv /tmp/prep_breakdown_pid*.json "$OUT/" 2>/dev/null

echo ""
echo "===== metrics ====="
grep -E "Avg Tokens per step \(incl|Avg Fraction|Avg Phase|Avg Cache|Total Throughput|Avg target verify|Avg draft step" "$OUT/run.log" | head -8

echo ""
echo "===== prep breakdown ====="
"$PY" -c "
import json, glob, statistics as st
files = sorted(glob.glob('$OUT/prep_breakdown_pid*.json'))
if not files:
    print('NO breakdown files'); exit()
d = json.load(open(files[-1]))
print(f'{\"key\":<40} {\"n\":>5} {\"med (μs)\":>10} {\"sum (ms)\":>9}')
for k in sorted(d.keys()):
    v = d[k]
    n = len(v)
    med = st.median(v) / 1000.0
    s = sum(v) / 1_000_000.0
    print(f'{k:<40} {n:>5} {med:>10.2f} {s:>9.2f}')
"
