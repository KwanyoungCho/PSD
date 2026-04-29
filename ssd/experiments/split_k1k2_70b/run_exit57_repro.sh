#!/bin/bash
# Reproduce exit_fine/exit_57 exactly: uniform, K1=K2=8, dfo=2, pfo=1, exit=57.
# Same CLI as run_exit_fine.sh's run() at L=57.
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
fi
echo "[gpu-pick] $CUDA_VISIBLE_DEVICES"
echo "[host-load] $(uptime | awk -F'load average:' '{print $2}')"

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python
TARGET=/data2/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data2/chokwans99/awq_calibrated/tinyllama_1b
TGT_ART=/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
DRAFT_ART=/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
OUT="$PWD/experiments/split_k1k2_70b/exit57_repro"
mkdir -p "$OUT"

COMMON="--llama --size 8 --model_path $TARGET --draft_path $DRAFT --async --spec --gpus 5 --b 1 --temp 0.6 --output_len 64 --max_model_len 2048 --random --numseqs 10"
QUANT="--quant_awq --quant_awq_artifact $TGT_ART --quant_group_size 128 --quant_awq_draft --quant_awq_draft_artifact $DRAFT_ART"

echo "[$(date +%H:%M:%S)] === exit_layer=57 (uniform, repro) ==="
fuser -k ${SSD_DIST_PORT}/tcp 2>/dev/null; sleep 2
SSD_PROFILE_DIR="$OUT" \
    "$PY" -O bench/bench.py $COMMON $QUANT \
    --k 16 --f 3 --mesa --mesa_exit_layer 57 \
    --mesa_draft_fan_out 2 --mesa_phase1_k 8 --mesa_phase2_k 8 \
    >"$OUT/run.log" 2>&1
rc=$?
tp=$(grep -oP 'Total Throughput:\s*\K[\d.]+' "$OUT/run.log" | head -1)
err=$(grep -oP 'Traceback|RuntimeError|FAIL|Aborted' "$OUT/run.log" | head -1)
echo "  rc=$rc TP=${tp:-?} ${err:+ERR=$err}"
echo "[host-load-after] $(uptime | awk -F'load average:' '{print $2}')"

"$PY" bench/plot_mesa_timeline.py "$OUT" --step 50 --warmup 5 >/dev/null 2>&1
"$PY" bench/plot_mesa_breakdown.py "$OUT" >/dev/null 2>&1

echo ""
echo "===== metrics ====="
grep -E "MESA split-K1K2.*layouts|Total Throughput|Avg Tokens per step|Avg Fraction|Avg Cache|Avg Phase|Avg draft step|Avg target verify|Avg target time|Avg Tokens per step on" "$OUT/run.log" | head -12

echo ""
echo "===== timing comparison vs Apr 28 exit_fine/exit_57 ====="
"$PY" -c "
import json, glob, statistics
def stats(path, label):
    rows = json.load(open(path))
    evts = [r['ms'] for r in rows if r['label']==label]
    return (len(evts), statistics.median(evts) if evts else 0)

old_t = '/home/chokwans99/PSD/ssd/experiments/split_k1k2_70b/exit_fine/exit_57/mesa_profile_target_rank0_203446.json'
old_d = '/home/chokwans99/PSD/ssd/experiments/split_k1k2_70b/exit_fine/exit_57/mesa_profile_draft_203446.json'
new_t = sorted(glob.glob('$OUT/mesa_profile_target_rank0_*.json'))[-1]
new_d = sorted(glob.glob('$OUT/mesa_profile_draft_*.json'))[-1]

print(f'{\"event\":<22} {\"Apr 28\":>10} {\"now\":>10} {\"ratio\":>8}')
for label in ['graph_pre', 'exit_logits', 'proxy_compute_send']:
    n_o, m_o = stats(old_t, label)
    n_n, m_n = stats(new_t, label)
    r = m_n/m_o if m_o else 0
    print(f'{label:<22} {m_o:>10.3f} {m_n:>10.3f} {r:>8.2f}x')
for label in ['glue', 'phase1_build', 'phase1_replay', 'proxy_wait', 'phase2_build', 'phase2_replay']:
    n_o, m_o = stats(old_d, label)
    n_n, m_n = stats(new_d, label)
    r = m_n/m_o if m_o else 0
    print(f'{label:<22} {m_o:>10.3f} {m_n:>10.3f} {r:>8.2f}x')
"
