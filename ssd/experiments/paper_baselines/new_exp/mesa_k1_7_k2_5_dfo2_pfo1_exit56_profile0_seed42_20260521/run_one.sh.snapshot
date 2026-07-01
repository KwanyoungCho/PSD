#!/usr/bin/env bash
# K2=5 paper baseline TPS verification — single run.
# Usage:  bash run_one.sh <label> <PROFILE_MESA>
# Sample: bash run_one.sh B_current_off 0
#
# Reproduces the exact 20260512_ours_label_perf_k1_7_k2_5 paper baseline
# command, only flipping the SSD_PROFILE_MESA env. SSD_PROFILE_DIR is
# unset when PROFILE_MESA=0 (cold path; no profile dump).

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/k2_5_tps_verify"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

LABEL="${1:?label arg (e.g., B_current_off) required}"
PROFILE="${2:?PROFILE_MESA arg (0 or 1) required}"
SEED="${3:-42}"

OUTDIR="${PHASE_DIR}/${LABEL}"
mkdir -p "${OUTDIR}"

cd "${ROOT}"

# Common env (verbatim from 20260512_ours_label_perf metadata).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12660
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_MESA="${PROFILE}"

if [[ "${PROFILE}" == "1" ]]; then
  export SSD_PROFILE_DIR="${OUTDIR}"
  export SSD_PROFILE_MESA_DETAIL=0
fi

echo "[$(date -Is)] === START ${LABEL} (PROFILE_MESA=${PROFILE}, seed=${SEED}) ===" | tee "${OUTDIR}/run.log"
pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# Paper K2=5 args (verbatim from 20260512_ours_label_perf metadata).
"${PY}" -O bench/bench.py \
  --llama --size 8 \
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
  --quant_awq \
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
  --quant_group_size 128 \
  --gpus 5 --b 1 --temp 0.7 --seed "${SEED}" --numseqs 50 \
  --input_len 512 --output_len 512 --all --max_model_len 2048 \
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
  --quant_awq_draft \
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
  --async --spec --k 12 --f 3 \
  --mesa --mesa_exit_layer 56 --mesa_phase1_k 7 --mesa_phase2_k 5 \
  --mesa_draft_fan_out 2 --mesa_policy b \
  2>&1 | tee -a "${OUTDIR}/run.log"

echo "[$(date -Is)] === END ${LABEL} ===" | tee -a "${OUTDIR}/run.log"
echo ""
echo "Decode TPS:" | tee -a "${OUTDIR}/run.log"
grep -E "Final Decode Throughput|Avg target time" "${OUTDIR}/run.log" | tee -a "${OUTDIR}/run.log"
