#!/bin/bash
# AR (no-speculation) and EAGLE comparison to accompany the DUET sweep.
# Usage: bash bench/run_ar_eagle.sh
#
# AR baselines:
#   - ar_layerskip: autoregressive on LayerSkip-Llama3-8B (matches DUET/baseline_f* target)
#   - ar_llama31:   autoregressive on Llama-3.1-8B-Instruct (matches EAGLE target)
#
# EAGLE (EAGLE-3 draft tuned for Llama-3.1-8B-Instruct):
#   - eagle_f3_k4: matched knobs to DUET (K=4, async_fan_out=3)

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/tmp
export SSD_EAGLE3_8B=/data2/chokwans99/models/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python

LAYERSKIP=/data2/chokwans99/models/layerskip-llama3-8B
LLAMA31=/data2/chokwans99/models/models--meta-llama--Llama-3.1-8B-Instruct

NUMSEQS=300
OUTLEN=256

OUTDIR="/tmp/ar_eagle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "Output dir: $OUTDIR"

SLOT_GPUS=("0,1" "2,3" "4,5" "6,7")
SLOT_PORTS=(12230 12240 12250 12260)
NUM_SLOTS=${#SLOT_GPUS[@]}
declare -a SLOT_PID
for ((i=0; i<NUM_SLOTS; i++)); do SLOT_PID[$i]=0; done

wait_for_slot() {
    while true; do
        for ((i=0; i<NUM_SLOTS; i++)); do
            local pid=${SLOT_PID[$i]}
            if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
                SLOT_PID[$i]=0; echo "$i"; return
            fi
        done
        sleep 3
    done
}

launch() {
    local label="$1"; shift
    local slot; slot=$(wait_for_slot)
    local gpus=${SLOT_GPUS[$slot]}; local port=${SLOT_PORTS[$slot]}
    local outfile="$OUTDIR/${label}.log"
    echo "[slot=$slot GPUs=$gpus port=$port] Launching: $label  $*"
    (
        export CUDA_VISIBLE_DEVICES="$gpus"
        export SSD_DIST_PORT="$port"
        "$PY" -O bench/bench.py "$@" >"$outfile" 2>&1
    ) &
    SLOT_PID[$slot]=$!
}

wait_all() {
    for ((i=0; i<NUM_SLOTS; i++)); do
        local pid=${SLOT_PID[$i]}
        if [[ $pid -ne 0 ]]; then wait "$pid" 2>/dev/null || true; SLOT_PID[$i]=0; fi
    done
}

# Autoregressive (no --spec / --async). --gpus 2 preserves the same TP setup.
AR_COMMON="--llama --size 8 --gpus 2 --b 1 --temp 0.6 --numseqs $NUMSEQS --output_len $OUTLEN --random"

launch "ar_layerskip" $AR_COMMON --model_path "$LAYERSKIP"
launch "ar_llama31"   $AR_COMMON --model_path "$LLAMA31"

# EAGLE on Llama-3.1-8B-Instruct with the same K/f as DUET's best config for apples-to-apples.
# EAGLE draft (1-layer) uses a different graph than Llama-3.2-1B; we match K=4, f=3.
EAGLE_COMMON="--llama --size 8 --eagle --async --k 4 --f 3 --gpus 2 --b 1 --temp 0.6 --numseqs $NUMSEQS --output_len $OUTLEN --random --model_path $LLAMA31"
launch "eagle_f3_k4" $EAGLE_COMMON

wait_all
echo ""
echo "=== AR+EAGLE SUMMARY ==="
printf "%-22s %-10s %-8s %-8s %-8s %-10s %-10s\n" "label" "TP" "CH" "TS" "AR" "Draft" "Verify"
for f in "$OUTDIR"/*.log; do
    label=$(basename "$f" .log)
    tp=$(grep "Total Throughput" "$f" | grep -oP 'Total Throughput:\s*\K[\d.]+' | head -1)
    ch=$(grep "Avg Cache Hits" "$f" | grep -oP '[\d.]+' | tail -1)
    ts=$(grep "Avg Tokens per step" "$f" | head -1 | grep -oP '[\d.]+' | tail -1)
    ar=$(grep "Avg Fraction" "$f" | grep -oP '[\d.]+' | tail -1)
    ds=$(grep "Avg draft step" "$f" | grep -oP '[\d.]+' | tail -1)
    tv=$(grep "Avg target verify" "$f" | grep -oP '[\d.]+' | tail -1)
    printf "%-22s %-10s %-8s %-8s %-8s %-10s %-10s\n" "$label" "${tp:-?}" "${ch:-?}" "${ts:-?}" "${ar:-?}" "${ds:-?}ms" "${tv:-?}ms"
done
