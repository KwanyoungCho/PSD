#!/usr/bin/env bash
# Batch 3v — 70B K1=K2=7 production shape with PROFILE_DUET_DETAIL=1.
# Three runs to verify H1 (proxy_compute_send = peer-wait) hypothesis
# AND to measure actual TPS impact when target verify is the bulk of
# wall time.
#
#   off_off  baseline (blocking send)
#   on_off   AsyncSendRing only
#   on_on    AsyncSendRing + proxy_stream
#
# Uses the same K1=K2=7 exit=52 dfo=2 pfo=1 shape as the earlier breakdown.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/batch3v_70b"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

# GPUs 3-7 free per earlier check; 5 GPUs (TP=4 target + 1 draft)
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export SSD_DIST_PORT=12671
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_DUET=1
export SSD_PROFILE_DUET_DETAIL=1

COMMON_ARGS=(
  --llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 20
  --input_len 512 --output_len 256 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --async --spec --k 14 --f 3
  --duet --duet_exit_layer 52 --duet_phase1_k 7 --duet_phase2_k 7
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
  sleep 5

  SSD_PROFILE_DIR="${outdir}" \
  SSD_ASYNC_PROXY_SEND="${async_send}" \
  SSD_PROXY_STREAM="${proxy_stream}" \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
    > "${outdir}/run.log" 2>&1

  echo "[$(date -Is)] === END ${label} ==="
}

run_one "off_off" 0 0   # baseline (current default)
run_one "on_off"  1 0   # async send only
run_one "on_on"   1 1   # full overlap

echo ""
echo "=== TPS + STEP COMPARISON ==="
for combo in "off_off" "on_off" "on_on"; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${combo}/run.log" | tail -1)
  step=$(grep "Avg target time per full step" "${PHASE_DIR}/${combo}/run.log" | tail -1)
  echo "  ${combo}: ${tps} | ${step}"
done

echo ""
echo "=== DUET profile paths ==="
ls -la "${PHASE_DIR}"/*/duet_profile_target_rank0_*.json 2>&1 || true
ls -la "${PHASE_DIR}"/*/proxy_send_ring_stats.json 2>&1 || true
