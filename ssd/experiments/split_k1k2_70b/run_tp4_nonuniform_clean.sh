#!/bin/bash
# Re-run nonuniform [5,3,1,1,0,0,0,0,5] @ exit=57, temp=0.6 on clean PXB GPUs 0-4.
# Same config as the original tp4_compare/nonuniform but on a same-PXB GPU set.
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

# Match original tp4_compare exactly: temp=0.6, n=10
COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

OUT="$PWD/experiments/split_k1k2_70b/tp4_nonuniform_clean"
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] === nonuniform (p1=5,3,1,1,0,0,0,0,5, temp=0.6, n=10, GPUs=$CUDA_VISIBLE_DEVICES) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
SSD_PROFILE_DIR="$OUT" \
    "$PY" -O bench/bench.py $COMMON $QUANT \
    --k 16 --f 3 --mesa --mesa_exit_layer 57 \
    --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 8 \
    --mesa_split_phase1_fan_out_list "5,3,1,1,0,0,0,0,5" \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
err=$(grep -oP 'Traceback|RuntimeError|Aborted' "$OUT/run.log" | head -1)
echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"

"$PY" bench/plot_mesa_timeline.py "$OUT" --step 50 --warmup 5 >/dev/null 2>&1
"$PY" bench/plot_mesa_breakdown.py "$OUT" >/dev/null 2>&1

echo ""
echo "===== metrics ====="
grep -E "MESA split-K1K2.*layouts|Total Throughput|Avg Tokens per step|Avg Fraction|Avg Cache|Avg Phase|Avg draft step|Avg target verify|Avg target time|Avg Tokens per step on" "$OUT/run.log" | head -12

echo ""
echo "===== graph_pre stats ====="
"$PY" -c "
import json, glob, statistics
ps = sorted(glob.glob('$OUT/mesa_profile_target_rank0_*.json'))
rows = json.load(open(ps[-1]))
for label in ['graph_pre', 'exit_logits', 'proxy_compute_send']:
    evts = [r['ms'] for r in rows if r['label']==label]
    if evts:
        print(f'  {label:<22} med={statistics.median(evts):.3f} min={min(evts):.3f} max={max(evts):.3f}')
ps2 = sorted(glob.glob('$OUT/mesa_profile_draft_*.json'))
rows2 = json.load(open(ps2[-1]))
for label in ['phase1_replay', 'phase2_replay']:
    evts = [r['ms'] for r in rows2 if r['label']==label]
    if evts:
        print(f'  {label:<22} med={statistics.median(evts):.3f} min={min(evts):.3f} max={max(evts):.3f}')
"
