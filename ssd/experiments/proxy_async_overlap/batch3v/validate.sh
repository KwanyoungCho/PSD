#!/usr/bin/env bash
# Batch 3v validation: correctness + perf for the AsyncSendRing + proxy_stream
# combination. 8B model, split-K1/K2 (K1=3 K2=2), greedy temp=0, 8 prompts.
#
# Three runs with all-else-equal:
#   (0,0)  baseline (blocking send, default stream)
#   (1,0)  async send only (ring buffer + isend)
#   (1,1)  async send + proxy_stream overlap
#
# Correctness: generated token sequences must be byte-identical across the
# three runs. Silent corruption from a missed record_stream would surface
# here.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/batch3v"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"
source "${ROOT}/env.sh" > /dev/null

export CUDA_VISIBLE_DEVICES=3,4,5
export SSD_DIST_PORT=12670
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1

mkdir -p "${PHASE_DIR}"

COMMON_ARGS=(
  --llama --size 8
  --gpus 3 --b 1 --temp 0 --seed 42 --numseqs 8
  --input_len 128 --output_len 128 --max_model_len 2048
  --async --spec --k 5 --f 3
  --duet --duet_exit_layer 21 --duet_phase1_k 3 --duet_phase2_k 2
  --duet_draft_fan_out 2 --duet_policy b
)

run_one() {
  local label="$1"
  local async_send="$2"
  local proxy_stream="$3"
  local outdir="${PHASE_DIR}/${label}"
  mkdir -p "${outdir}"

  echo "[$(date -Is)] === START ${label} (ASYNC=${async_send} STREAM=${proxy_stream}) ==="
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 3

  SSD_ASYNC_PROXY_SEND="${async_send}" \
  SSD_PROXY_STREAM="${proxy_stream}" \
  SSD_PROFILE_MESA=0 SSD_PROFILE_DUET=0 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
    > "${outdir}/run.log" 2>&1

  echo "[$(date -Is)] === END ${label} ==="
}

run_one "off_off" 0 0   # baseline
run_one "on_off"  1 0   # async send only
run_one "on_on"   1 1   # async send + proxy_stream

echo ""
echo "=== BYTE-IDENTICAL GENERATION CHECK ==="
for combo in "on_off" "on_on"; do
  diff_out=$(diff <(grep "^Generation:" "${PHASE_DIR}/off_off/run.log") \
                   <(grep "^Generation:" "${PHASE_DIR}/${combo}/run.log") || true)
  if [ -z "${diff_out}" ]; then
    echo "  off_off vs ${combo}:  byte-identical ✓"
  else
    echo "  off_off vs ${combo}:  DIFFERS ✗"
    echo "${diff_out}" | head -20
  fi
done

echo ""
echo "=== TPS COMPARISON ==="
for combo in "off_off" "on_off" "on_on"; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${combo}/run.log" | tail -1)
  step=$(grep "Avg target time per full step" "${PHASE_DIR}/${combo}/run.log" | tail -1)
  echo "  ${combo}: ${tps} | ${step}"
done
