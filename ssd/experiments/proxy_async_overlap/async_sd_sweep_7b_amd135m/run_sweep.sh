#!/usr/bin/env bash
# Async SD (no MESA) sweep for the 7B + AMD-135m baseline.
# K = 7..10 × F = 3..10  → 32 runs total.
# 4-way parallel: each slot uses 2 GPUs (target + draft) and its own port range.
#   slot 0: GPUs 0,1   port base 12700
#   slot 1: GPUs 2,3   port base 12800
#   slot 2: GPUs 4,5   port base 12900
#   slot 3: GPUs 6,7   port base 13000
# Jobs are round-robin assigned across slots so K and F load are balanced.
set -uo pipefail

ROOT=/home/chokwans99/PSD/ssd
SWEEP_DIR=$ROOT/experiments/proxy_async_overlap/async_sd_sweep_7b_amd135m
RUN_ONE=$SWEEP_DIR/run_one.sh

declare -a SLOT_GPUS=(  "0,1" "2,3" "4,5" "6,7" )
declare -a SLOT_PORTBASE=( 12700 12800 12900 13000 )

# Build job list (K F) in natural order.
declare -a JOBS=()
for K in 7 8 9 10; do
  for F in 3 4 5 6 7 8 9 10; do
    JOBS+=("$K $F")
  done
done

# Partition jobs across 4 slots round-robin.
declare -a SLOT_JOBS_0 SLOT_JOBS_1 SLOT_JOBS_2 SLOT_JOBS_3
for i in "${!JOBS[@]}"; do
  s=$(( i % 4 ))
  case "$s" in
    0) SLOT_JOBS_0+=("${JOBS[$i]}") ;;
    1) SLOT_JOBS_1+=("${JOBS[$i]}") ;;
    2) SLOT_JOBS_2+=("${JOBS[$i]}") ;;
    3) SLOT_JOBS_3+=("${JOBS[$i]}") ;;
  esac
done

run_slot() {
  local slot=$1
  local gpus="${SLOT_GPUS[$slot]}"
  local portbase=${SLOT_PORTBASE[$slot]}
  shift
  for job in "$@"; do
    read -r K F <<< "$job"
    local port=$(( portbase + K * 10 + F ))
    echo "[slot ${slot}] starting k=${K} f=${F} gpus=${gpus} port=${port}"
    bash "$RUN_ONE" "$K" "$F" "$gpus" "$port"
    echo "[slot ${slot}] finished k=${K} f=${F}"
  done
}

echo "[$(date -Is)] === SWEEP START (32 runs, 4-way parallel) ==="
echo "slot 0 jobs (${#SLOT_JOBS_0[@]}): ${SLOT_JOBS_0[*]}"
echo "slot 1 jobs (${#SLOT_JOBS_1[@]}): ${SLOT_JOBS_1[*]}"
echo "slot 2 jobs (${#SLOT_JOBS_2[@]}): ${SLOT_JOBS_2[*]}"
echo "slot 3 jobs (${#SLOT_JOBS_3[@]}): ${SLOT_JOBS_3[*]}"

run_slot 0 "${SLOT_JOBS_0[@]}" &> "$SWEEP_DIR/slot0.log" &
PID0=$!
run_slot 1 "${SLOT_JOBS_1[@]}" &> "$SWEEP_DIR/slot1.log" &
PID1=$!
run_slot 2 "${SLOT_JOBS_2[@]}" &> "$SWEEP_DIR/slot2.log" &
PID2=$!
run_slot 3 "${SLOT_JOBS_3[@]}" &> "$SWEEP_DIR/slot3.log" &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3
echo "[$(date -Is)] === SWEEP COMPLETE ==="

echo ""
echo "=== Headline summary ==="
for K in 7 8 9 10; do
  for F in 3 4 5 6 7 8 9 10; do
    OUTDIR="${SWEEP_DIR}/k${K}_f${F}"
    echo "--- k=${K} f=${F} ---"
    cat "${OUTDIR}/headline.txt" 2>/dev/null || echo "(no headline)"
  done
done
