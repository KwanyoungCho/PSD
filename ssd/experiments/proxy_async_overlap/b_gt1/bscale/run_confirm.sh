#!/usr/bin/env bash
# bscale confirm phase — B=8 scan winner vs C_b8, 3-rep INTERLEAVED
# (D,C,D,C,D,C), ns=20 out=256 in=512 seed 42 temp 0.7, GPUs 0-4.
# Winner config via env: B8_K1 B8_K2 B8_DFO B8_PFO.
# Optional B=4 edge confirm (only if a scan edge cell beat 165.5):
# set B4_CONFIRM=1 plus B4_K1 B4_K2 B4_DFO B4_PFO.
# Ports ascend from 13000.
set -uo pipefail
: "${B8_K1:?}" "${B8_K2:?}" "${B8_DFO:?}" "${B8_PFO:?}"
B4_CONFIRM="${B4_CONFIRM:-0}"
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/bscale/confirm"
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

PORT=13000
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

echo "[$(date -Is)] confirm winner: B8 K1=${B8_K1} K2=${B8_K2} dfo=${B8_DFO} pfo=${B8_PFO}; B4_CONFIRM=${B4_CONFIRM}"

for rep in 1 2 3; do
  duet_run "b8_duet_r${rep}" 8 "${B8_K1}" "${B8_K2}" "${B8_DFO}" "${B8_PFO}"
  run_one "b8_c_r${rep}" --b 8 --k 7 --f 6
  PORT=$((PORT + 1))
done

if [[ "${B4_CONFIRM}" == "1" ]]; then
  : "${B4_K1:?}" "${B4_K2:?}" "${B4_DFO:?}" "${B4_PFO:?}"
  for rep in 1 2 3; do
    duet_run "b4_duet_r${rep}" 4 "${B4_K1}" "${B4_K2}" "${B4_DFO}" "${B4_PFO}"
    run_one "b4_c_r${rep}" --b 4 --k 7 --f 6
    PORT=$((PORT + 1))
  done
fi

echo "[$(date -Is)] === BSCALE CONFIRM DONE ==="
