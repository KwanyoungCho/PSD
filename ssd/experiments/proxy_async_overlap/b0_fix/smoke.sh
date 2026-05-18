#!/usr/bin/env bash
# Smoke test for the B=0 crash guard (SpecDecodeStep.decode early return).
#
# Reproduces the formerly-fastest-crashing cell from commit 501cab4:
#   (dfo=5, pfo=3, f=8) crashed in ~2 min on first MESA step.
# With the guard, the run should complete normally (or at least pass the
# point that previously crashed).

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/b0_fix/smoke"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12655
export SSD_PROFILE_MESA=0
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1

mkdir -p "${PHASE_DIR}"
pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# Smaller run for fast smoke (ns=10, out=128) but high f to stress
# the scheduler preemption pattern.
DFO=5
PFO=3
F=$((DFO + PFO))

echo "[$(date -Is)] === SMOKE (dfo=${DFO}, pfo=${PFO}, f=${F}) ==="
"${PY}" -O bench/bench.py \
  --llama --size 8 \
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
  --quant_awq \
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
  --quant_group_size 128 \
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 10 \
  --input_len 512 --output_len 128 --all --max_model_len 2048 \
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
  --quant_awq_draft \
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
  --async --spec --k 14 --f ${F} \
  --mesa --mesa_exit_layer 52 --mesa_phase1_k 7 --mesa_phase2_k 7 \
  --mesa_draft_fan_out ${DFO} --mesa_policy b \
  2>&1 | tee "${PHASE_DIR}/run.log"

echo "[$(date -Is)] === SMOKE DONE ==="
