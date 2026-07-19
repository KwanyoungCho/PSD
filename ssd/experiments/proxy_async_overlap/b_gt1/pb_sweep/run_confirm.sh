#!/usr/bin/env bash
# pb_sweep confirm phase — top scan config per B vs C, 3-rep INTERLEAVED
# (D,C,D,C,D,C per B), ns=20 out=256 in=512 seed 42 temp 0.7, GPUs 0-4.
# Winner configs are passed via env:
#   B4_K1 B4_K2 B4_DFO B4_PFO  and  B2_K1 B2_K2 B2_DFO B2_PFO
# Ports continue ascending from 12950.
set -uo pipefail
: "${B4_K1:?}" "${B4_K2:?}" "${B4_DFO:?}" "${B4_PFO:?}"
: "${B2_K1:?}" "${B2_K2:?}" "${B2_DFO:?}" "${B2_PFO:?}"
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/pb_sweep/confirm"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_PROFILE_DUET=0

BASE_ARGS=(
  --llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128
  --gpus 5 --temp 0.7 --seed 42 --numseqs 20
  --input_len 512 --output_len 256 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --async --spec
)

cleanup() {
  pkill -9 -u chokwans99 -f "bench/bench.py" 2>/dev/null || true
  pkill -9 -u chokwans99 -f "multiprocessing.spawn import spawn_main" 2>/dev/null || true
  pkill -9 -u chokwans99 -f "multiprocessing.resource_tracker" 2>/dev/null || true
  sleep 6
}

PORT=12950
run_one() {
  local label="$1"; shift
  local outdir="${OUT}/${label}"
  mkdir -p "${outdir}"
  cleanup
  echo "[$(date -Is)] === START ${label} (port ${PORT}) ==="
  SSD_DIST_PORT="${PORT}" timeout -k 30 1800 \
    "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1
  local rc=$?
  PORT=$((PORT + 1))
  cleanup
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || true)
  echo "[$(date -Is)] === END ${label} rc=${rc}: ${tps:-NO_TPS} ==="
}

duet_run() {  # duet_run <label> <B> <K1> <K2> <dfo> <pfo>
  local label="$1" B="$2" K1="$3" K2="$4" dfo="$5" pfo="$6"
  local k=$((K1 + K2)) f=$((dfo + pfo))
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "${label}" --b "${B}" \
      --k "${k}" --f "${f}" --duet --duet_exit_layer 56 \
      --duet_phase1_k "${K1}" --duet_phase2_k "${K2}" \
      --duet_draft_fan_out "${dfo}" --duet_policy b )
  PORT=$((PORT + 1))
}

echo "[$(date -Is)] confirm winners: B4 K1=${B4_K1} K2=${B4_K2} dfo=${B4_DFO} pfo=${B4_PFO}; B2 K1=${B2_K1} K2=${B2_K2} dfo=${B2_DFO} pfo=${B2_PFO}"

for rep in 1 2 3; do
  duet_run "b4_duet_r${rep}" 4 "${B4_K1}" "${B4_K2}" "${B4_DFO}" "${B4_PFO}"
  run_one "b4_c_r${rep}" --b 4 --k 7 --f 6
  PORT=$((PORT + 1))
done
for rep in 1 2 3; do
  duet_run "b2_duet_r${rep}" 2 "${B2_K1}" "${B2_K2}" "${B2_DFO}" "${B2_PFO}"
  run_one "b2_c_r${rep}" --b 2 --k 7 --f 6
  PORT=$((PORT + 1))
done

echo "[$(date -Is)] === PB_SWEEP CONFIRM DONE ==="
