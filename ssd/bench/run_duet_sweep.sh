#!/bin/bash
# DUET-SSD parameter sweep: exit_layer × fan_out × phase1/phase2 split
# Usage: bash bench/run_duet_sweep.sh        (uses all 8 GPUs, 4-way parallel, 2 GPUs per run)
#
# Dimensions:
#   1. Total draft budget: MQ_LEN = async_fan_out × (K+1). Sweep --f in {2,3,4,5} with K=4.
#   2. Phase1/Phase2 split: --duet_draft_fan_out (Phase1 per-position branches).
#      Phase2 (proxy) gets async_fan_out - draft_fan_out.
#   3. Exit layer: at (f=3, dfo=1) sweep {10,16,21,26}/32 (2/3 of L is the default).
#
# Baseline SSD uses the same --f so total budget matches DUET for fair comparison.

set -u
cd "$(dirname "$0")/.."   # ssd repo root — so `import ssd.paths` resolves
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SSD_HF_CACHE=/data2/chokwans99/models
export SSD_CUDA_ARCH=8.6       # RTX 3090 -> sm_86
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_DATASET_DIR=/tmp

PY=/home/chokwans99/anaconda3/envs/ssd/bin/python

TARGET_MODEL=/data2/chokwans99/models/layerskip-llama3-8B
DRAFT_MODEL=/data2/chokwans99/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots

NUMSEQS=300     # dataset files absent locally → --random; 300 samples for statistical reliability
OUTLEN=256
COMMON="--llama --size 8 --model_path $TARGET_MODEL --draft_path $DRAFT_MODEL --async --spec --k 4 --gpus 2 --b 1 --temp 0.6 --numseqs $NUMSEQS --output_len $OUTLEN --random"

OUTDIR="/tmp/duet_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "Output dir: $OUTDIR"

# Slots: (GPUs, nccl port). 4 concurrent jobs on 8 GPUs.
SLOT_GPUS=("0,1" "2,3" "4,5" "6,7")
SLOT_PORTS=(12230 12240 12250 12260)
NUM_SLOTS=${#SLOT_GPUS[@]}

declare -a SLOT_PID    # pid occupying each slot (0 = free)
for ((i=0; i<NUM_SLOTS; i++)); do SLOT_PID[$i]=0; done

wait_for_slot() {
    # Return the index of the first free slot, waiting if needed.
    while true; do
        for ((i=0; i<NUM_SLOTS; i++)); do
            local pid=${SLOT_PID[$i]}
            if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
                SLOT_PID[$i]=0
                echo "$i"
                return
            fi
        done
        sleep 3
    done
}

launch() {
    local label="$1"; shift
    local slot; slot=$(wait_for_slot)
    local gpus=${SLOT_GPUS[$slot]}
    local port=${SLOT_PORTS[$slot]}
    local outfile="$OUTDIR/${label}.log"
    echo ""
    echo "[slot=$slot GPUs=$gpus port=$port] Launching: $label  $*"
    (
        export CUDA_VISIBLE_DEVICES="$gpus"
        export SSD_DIST_PORT="$port"
        "$PY" -O bench/bench.py $COMMON "$@" >"$outfile" 2>&1
    ) &
    SLOT_PID[$slot]=$!
}

wait_all() {
    for ((i=0; i<NUM_SLOTS; i++)); do
        local pid=${SLOT_PID[$i]}
        if [[ $pid -ne 0 ]]; then wait "$pid" 2>/dev/null || true; SLOT_PID[$i]=0; fi
    done
}

# ---------- Phase A: baseline & DUET f/split sweep (exit=21 fixed) ----------
# Baselines at matched budgets (MQ_LEN = f * (K+1) = f * 5)
launch "baseline_f2" --f 2
launch "baseline_f3" --f 3
launch "baseline_f4" --f 4
launch "baseline_f5" --f 5

# DUET f=2: (1,1)
launch "duet_f2_dfo1" --f 2 --duet --duet_exit_layer 21 --duet_draft_fan_out 1

# DUET f=3: (1,2) (2,1)
launch "duet_f3_dfo1" --f 3 --duet --duet_exit_layer 21 --duet_draft_fan_out 1
launch "duet_f3_dfo2" --f 3 --duet --duet_exit_layer 21 --duet_draft_fan_out 2

# DUET f=4: (1,3) (2,2) (3,1)
launch "duet_f4_dfo1" --f 4 --duet --duet_exit_layer 21 --duet_draft_fan_out 1
launch "duet_f4_dfo2" --f 4 --duet --duet_exit_layer 21 --duet_draft_fan_out 2
launch "duet_f4_dfo3" --f 4 --duet --duet_exit_layer 21 --duet_draft_fan_out 3

# DUET f=5: (1,4) (2,3) (3,2) (4,1)
launch "duet_f5_dfo1" --f 5 --duet --duet_exit_layer 21 --duet_draft_fan_out 1
launch "duet_f5_dfo2" --f 5 --duet --duet_exit_layer 21 --duet_draft_fan_out 2
launch "duet_f5_dfo3" --f 5 --duet --duet_exit_layer 21 --duet_draft_fan_out 3
launch "duet_f5_dfo4" --f 5 --duet --duet_exit_layer 21 --duet_draft_fan_out 4

# ---------- Phase B: exit_layer sweep at (f=3, dfo=1) ----------
launch "duet_f3_dfo1_exit10" --f 3 --duet --duet_exit_layer 10 --duet_draft_fan_out 1
launch "duet_f3_dfo1_exit16" --f 3 --duet --duet_exit_layer 16 --duet_draft_fan_out 1
launch "duet_f3_dfo1_exit26" --f 3 --duet --duet_exit_layer 26 --duet_draft_fan_out 1

wait_all
echo ""
echo "=========================================="
echo "All done. Results in $OUTDIR"
echo "=========================================="

# Summary
echo ""
echo "=== SUMMARY ==="
printf "%-28s %-10s %-8s %-8s %-8s %-10s %-10s\n" "label" "TP" "CH" "TS" "AR" "Draft" "Verify"
for f in "$OUTDIR"/*.log; do
    label=$(basename "$f" .log)
    tp=$(grep "Total Throughput" "$f" | grep -oP 'Total Throughput:\s*\K[\d.]+' | head -1)
    ch=$(grep "Avg Cache Hits" "$f" | grep -oP '[\d.]+' | tail -1)
    ts=$(grep "Avg Tokens per step" "$f" | head -1 | grep -oP '[\d.]+' | tail -1)
    ar=$(grep "Avg Fraction" "$f" | grep -oP '[\d.]+' | tail -1)
    ds=$(grep "Avg draft step" "$f" | grep -oP '[\d.]+' | tail -1)
    tv=$(grep "Avg target verify" "$f" | grep -oP '[\d.]+' | tail -1)
    printf "%-28s %-10s %-8s %-8s %-8s %-10s %-10s\n" "$label" "${tp:-?}" "${ch:-?}" "${ts:-?}" "${ar:-?}" "${ds:-?}ms" "${tv:-?}ms"
done
