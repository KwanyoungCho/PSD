#!/usr/bin/env bash
# Final-Sanity — short 70B K1=K2=7 overhead gate before the full attribution run.
#
# Goal: confirm Phase-B aligned trace (commit 0d3911f) costs ≤1% decode TPS
# under SSD_PROFILE_MESA=1 in the realistic 70B AWQ setting before we burn
# 15 min on the full attribution experiment.
#
# Baseline 2026-05-13 metadata, only reductions: --numseqs 20 --output_len 128.
# Both runs use identical args except SSD_PROFILE_MESA env.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/p99_attribution/sanity"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

# Common env (verbatim from 2026-05-13 baseline metadata).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12651
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1

# Common args — same as baseline except smaller (ns=20, out=128 for ~5 min/run).
COMMON_ARGS=(
  --llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 20
  --input_len 512 --output_len 128 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --async --spec --k 14 --f 3
  --mesa --mesa_exit_layer 56 --mesa_phase1_k 7 --mesa_phase2_k 7
  --mesa_draft_fan_out 2 --mesa_policy b
)

run_one() {
  local label="$1"
  local profile="$2"
  local outdir="${PHASE_DIR}/${label}"
  mkdir -p "${outdir}"

  echo "[$(date -Is)] === START ${label} (PROFILE_MESA=${profile}) ===" | tee "${outdir}/run.log"
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 5

  SSD_PROFILE_DIR="${outdir}" SSD_PROFILE_MESA="${profile}" \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "${outdir}/run.log"

  echo "[$(date -Is)] === END ${label} ===" | tee -a "${outdir}/run.log"
}

# A: PROFILE_MESA=0 (no anchor / context / row build cost)
run_one "off" 0

# B: PROFILE_MESA=1 (Phase-B aligned trace fully enabled)
run_one "on" 1

echo "[$(date -Is)] === Sanity complete ==="
echo "Compare:"
echo "  grep 'Final Decode' ${PHASE_DIR}/off/run.log"
echo "  grep 'Final Decode' ${PHASE_DIR}/on/run.log"
