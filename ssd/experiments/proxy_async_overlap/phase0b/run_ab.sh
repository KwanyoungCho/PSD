#!/usr/bin/env bash
# Phase 0b — A/B baseline measurement for MESA proxy async-overlap work.
#
# Goal: confirm DETAIL probe does not perturb proxy_compute_send outer timing,
# and decompose outer into compute/pack/send inner spans + unattributed stall.
#
# Both runs use the EXACT command from
#   experiments/paper_baselines/final_experiments/20260513_ours_k1_7_k2_7_*/
# except:
#   - SSD_PROFILE_DIR is per-run
#   - SSD_PROFILE_MESA_DETAIL flips 0 ↔ 1
#   - --output_len is reduced 512 → 256 for faster probe (identical A vs B)
#
# Everything else (--temp 0.7 --seed 42 --numseqs 50 --input_len 512 --all,
# AWQ paths, --k 14, --mesa_phase1_k 7 --mesa_phase2_k 7,
# --mesa_exit_layer 56, --mesa_draft_fan_out 2, --mesa_policy b,
# SSD_FORCE_SPLIT_K1K2=1) is preserved verbatim from baseline metadata.
#
# Runs A and B are EXECUTED SEQUENTIALLY (not concurrent) to avoid GPU
# contention biasing the measurement.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/phase0b"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

# Common env (verbatim from baseline metadata)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12643
export SSD_PROFILE_MESA=1
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1

# Common args (verbatim from baseline metadata, --output_len 256 for faster probe)
COMMON_ARGS=(
  --llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 50
  --input_len 512 --output_len 256 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --async --spec --k 14 --f 3
  --mesa --mesa_exit_layer 56 --mesa_phase1_k 7 --mesa_phase2_k 7
  --mesa_draft_fan_out 2 --mesa_policy b
)

run_one() {
  local label="$1"
  local detail="$2"
  local outdir="${PHASE_DIR}/${label}"
  mkdir -p "${outdir}"

  echo "[$(date -Is)] === START ${label} (DETAIL=${detail}) ===" | tee "${outdir}/run.log"

  # Sequential: kill any stragglers from prior runs to avoid GPU memory leak.
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 5

  SSD_PROFILE_DIR="${outdir}" SSD_PROFILE_MESA_DETAIL="${detail}" \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "${outdir}/run.log"

  echo "[$(date -Is)] === END ${label} ===" | tee -a "${outdir}/run.log"
}

# A: DETAIL=0  (outer-only; reproduces baseline 2.31ms)
run_one "A_detail0" 0

# B: DETAIL=1  (outer + inner decomposition; tests probe effect)
run_one "B_detail1" 1

echo "[$(date -Is)] === Phase 0b A/B complete ==="
echo "A logs:  ${PHASE_DIR}/A_detail0/run.log"
echo "B logs:  ${PHASE_DIR}/B_detail1/run.log"
echo "Profile JSONs: ${PHASE_DIR}/{A_detail0,B_detail1}/mesa_profile_*.json"
