#!/usr/bin/env bash
# Phase 0 — Clean baseline for 08-proxy-overlap-experiment.md.
#
# Establishes TPS_baseline against which Phase 2/3/5 gates will be measured.
# 3 repetitions, full env hygiene per reviewer feedback:
#   - SSD_PROFILE_MESA=0 / SSD_PROFILE_MESA_DETAIL=0  (no event recording)
#   - SSD_PROFILE_{DRAFT,TARGET,} unset                (no per-step timing)
#   - SSD_TRACE_{BUCKET,SPLIT_K1K2} unset              (no debug print)
#
# Same K1=K2=7 paper config used in the breakdown run; output_len=256 for
# the same step count as 20260513 baseline and prior phase-0b runs.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/phase0_baseline"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
N_REPS=3

cd "${ROOT}"

# Common env — strict profile-off
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1
export SSD_PROFILE_MESA=0
export SSD_PROFILE_MESA_DETAIL=0
# Explicitly unset all profile/trace envs in case they leaked from parent:
unset SSD_PROFILE_DRAFT SSD_PROFILE_TARGET SSD_PROFILE \
      SSD_TRACE_BUCKET SSD_TRACE_SPLIT_K1K2

# Common args — verbatim 20260513 baseline metadata
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
  --mesa --mesa_exit_layer 52 --mesa_phase1_k 7 --mesa_phase2_k 7
  --mesa_draft_fan_out 2 --mesa_policy b
)

run_one() {
  local rep="$1"
  local port="$2"
  local outdir="${PHASE_DIR}/rep_${rep}"
  mkdir -p "${outdir}"

  echo "[$(date -Is)] === START rep ${rep} (port ${port}) ==="
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 5

  SSD_DIST_PORT="${port}" SSD_PROFILE_DIR="${outdir}" \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
    > "${outdir}/run.log" 2>&1

  local tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1)
  local step=$(grep "Avg target time per full step" "${outdir}/run.log" | tail -1)
  echo "[$(date -Is)] === END rep ${rep} ==="
  echo "  ${tps}"
  echo "  ${step}"
}

for i in 1 2 3; do
  port=$((12700 + i))
  run_one "${i}" "${port}"
done

echo "[$(date -Is)] === Phase 0 baseline complete ==="
echo ""
echo "Summary:"
for i in 1 2 3; do
  printf "  rep %d : %s   %s\n" "${i}" \
    "$(grep 'Final Decode Throughput' ${PHASE_DIR}/rep_${i}/run.log | tail -1)" \
    "$(grep 'Avg target time per full step' ${PHASE_DIR}/rep_${i}/run.log | tail -1)"
done
