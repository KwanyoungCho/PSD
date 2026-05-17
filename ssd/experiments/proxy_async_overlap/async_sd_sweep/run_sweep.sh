#!/usr/bin/env bash
# Async SD (no MESA) sweep — k=7..10 × f=3..6 = 16 runs.
# All runs at SSD_PROFILE_MESA=0 (cold path → no measurement overhead).
# Config matches the 20260512_ours_label_perf paper baseline minus MESA:
#   70B AWQ TP=4 target + TinyLlama-1.1B AWQ TP=1 draft, ns=50 in=512 out=512,
#   --temp 0.7 --seed 42 --all --max_model_len 2048.
#
# Per-run subdirs:  k{K}_f{F}/run.log
# Continue-on-failure: a single OOM does not abort the rest of the sweep.

set -uo pipefail

ROOT="/home/chokwans99/PSD/ssd"
SWEEP_DIR="${ROOT}/experiments/proxy_async_overlap/async_sd_sweep"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

# Common env. NO MESA env vars; SSD_PROFILE_MESA unset == "0".
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT_BASE=12700
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib

run_one() {
  local K=$1 F=$2
  local OUTDIR="${SWEEP_DIR}/k${K}_f${F}"
  mkdir -p "${OUTDIR}"

  # Unique NCCL port per run so a stale leftover doesn't block the next.
  export SSD_DIST_PORT="$((SSD_DIST_PORT_BASE + K * 10 + F))"

  echo "[$(date -Is)] === START k=${K} f=${F} (port ${SSD_DIST_PORT}) ===" | tee "${OUTDIR}/run.log"
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 8

  "${PY}" -O bench/bench.py \
    --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --quant_awq \
    --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
    --quant_group_size 128 \
    --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 50 \
    --input_len 512 --output_len 512 --all --max_model_len 2048 \
    --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
    --quant_awq_draft \
    --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
    --async --spec --k "${K}" --f "${F}" \
    >> "${OUTDIR}/run.log" 2>&1
  local status=$?

  if [[ ${status} -ne 0 ]]; then
    echo "[$(date -Is)] === FAILED k=${K} f=${F} status=${status} ===" | tee -a "${OUTDIR}/run.log"
  else
    echo "[$(date -Is)] === END k=${K} f=${F} ===" | tee -a "${OUTDIR}/run.log"
  fi

  # Extract headline TPS into a per-run summary file (always, even on fail).
  {
    echo "k=${K} f=${F} status=${status}"
    grep -E "Final Decode Throughput|Avg target time|Avg Tokens per step|Avg Fraction of Speculated|Cache hit rate" "${OUTDIR}/run.log" || true
  } > "${OUTDIR}/headline.txt"

  return 0
}

echo "[$(date -Is)] === SWEEP START (16 runs) ==="
for K in 7 8 9 10; do
  for F in 3 4 5 6; do
    run_one "${K}" "${F}"
  done
done
echo "[$(date -Is)] === SWEEP COMPLETE ==="

echo ""
echo "=== Headline summary ==="
for K in 7 8 9 10; do
  for F in 3 4 5 6; do
    OUTDIR="${SWEEP_DIR}/k${K}_f${F}"
    echo "--- k=${K} f=${F} ---"
    cat "${OUTDIR}/headline.txt" 2>/dev/null || echo "(no headline)"
  done
done
