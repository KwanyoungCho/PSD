#!/usr/bin/env bash
# Final-Run — 70B K1=K2=7 PROFILE_MESA=1 full attribution dump.
#
# Produces Phase-B aligned JSON for target_spec_wait p99 attribution.
# Same args as 2026-05-13 baseline metadata: --numseqs 50 --output_len 256
# (output_len reduced 512→256 to match Phase 0b A/B; still ~270 spec steps
# per ns=50, plenty for stable p99 estimation).

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/p99_attribution/full"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12652
export SSD_PROFILE_MESA=1
export SSD_PROFILE_DIR="${PHASE_DIR}"
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1

mkdir -p "${PHASE_DIR}"
pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

echo "[$(date -Is)] === START full PROFILE_MESA=1 run ==="
"${PY}" -O bench/bench.py \
  --llama --size 8 \
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
  --quant_awq \
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
  --quant_group_size 128 \
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 50 \
  --input_len 512 --output_len 256 --all --max_model_len 2048 \
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
  --quant_awq_draft \
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
  --async --spec --k 14 --f 3 \
  --mesa --mesa_exit_layer 56 --mesa_phase1_k 7 --mesa_phase2_k 7 \
  --mesa_draft_fan_out 2 --mesa_policy b \
  2>&1 | tee "${PHASE_DIR}/run.log"

echo "[$(date -Is)] === END full run ==="
ls -la "${PHASE_DIR}"/mesa_profile_*.json
