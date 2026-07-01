#!/usr/bin/env bash
# MESA K1=7 K2=5 PROFILE_MESA=1 run, paired with the PROFILE=0 cells in
# this same new_exp directory. SSD_PROFILE_DIR dumps per-event JSON here.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
OUTDIR="${ROOT}/experiments/paper_baselines/new_exp/mesa_k1_7_k2_5_dfo2_pfo1_exit56_profile1_seed42_20260521"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

SEED="${1:-42}"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12674
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_MESA=1
export SSD_PROFILE_DIR="${OUTDIR}"
export SSD_PROFILE_MESA_DETAIL=0

echo "[$(date -Is)] === START MESA K1=7 K2=5 PROFILE_MESA=1 (seed=${SEED}) ===" | tee "${OUTDIR}/run.log"
pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

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

echo "[$(date -Is)] === END MESA PROFILE=1 (seed=${SEED}) ===" | tee -a "${OUTDIR}/run.log"

{
  echo "seed=${SEED} profile=1"
  grep -E "Final Decode Throughput|Avg target time|Avg target verify|Avg Tokens per step|Avg Fraction|Avg Cache Hits|Avg Phase|Avg draft step" "${OUTDIR}/run.log" || true
} > "${OUTDIR}/headline.txt"

cat "${OUTDIR}/headline.txt"
