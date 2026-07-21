#!/usr/bin/env bash
# bscale32 Phase C — confirms, 3-rep interleaved D/C/D/C/D/C per B, out=256.
# 0. Probe db8_k1x1_d5p1 (ns=12): K1=1 was never scanned at B=8; if it beats
#    213.51 (b8_k2x2_d5p1 scan level) the B=8 DUET confirm shape becomes
#    k1x1_d5p1, else k2x2_d5p1.
# 1. Confirm pairs:
#    B=2  (ns=20): DUET k6x5_d3p1 vs C k5f6
#    B=4  (ns=20): DUET k3x3_d4p1 vs C k3f6
#    B=8  (ns=20): DUET (probe winner) vs C k3f6
#    B=16 (ns=16): DUET k1x1_d5p1 vs C k2f3
#    B=32 (ns=32): DUET k1x1_d4p1 vs C k2f2
# Crash policy: rc!=0 without OOM signature -> retry ONCE (first log kept as
# run.attempt1.log); OOM -> DNF, continue. Ports 13500+ (step 2 per attempt).
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/bscale32"
CONF="${OUT}/confirm32"
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
  --gpus 5 --temp 0.7 --seed 42 --numseqs 12
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

PORT=13500
LAST_RC=0
EXTRA_ENV=""

run_one() {  # run_one <base> <label> <bench args...>
  local base="$1" label="$2"; shift 2
  local outdir="${base}/${label}"
  mkdir -p "${outdir}"
  cleanup
  echo "[$(date -Is)] === START ${label} (port ${PORT}) ==="
  SSD_DIST_PORT="${PORT}" timeout -k 30 1800 \
    env ${EXTRA_ENV} "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1
  LAST_RC=$?
  PORT=$((PORT + 2))
  cleanup
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || true)
  echo "[$(date -Is)] === END ${label} rc=${LAST_RC}: ${tps:-NO_TPS} ==="
}

run_cell() {  # run_one + retry-once on non-OOM crash
  local base="$1" label="$2"; shift 2
  run_one "${base}" "${label}" "$@"
  if [[ ${LAST_RC} -ne 0 ]] && \
     ! grep -qE "OutOfMemoryError|CUDA out of memory" "${base}/${label}/run.log"; then
    echo "[$(date -Is)] === RETRY ${label} (rc=${LAST_RC}, no OOM signature) ==="
    mv "${base}/${label}/run.log" "${base}/${label}/run.attempt1.log"
    run_one "${base}" "${label}" "$@"
  fi
}

c_cell() {  # c_cell <base> <label> <B> <k> <f> <ns>
  local base="$1" label="$2" B="$3" k="$4" f="$5" ns="$6"
  EXTRA_ENV=""
  run_cell "${base}" "${label}" --b "${B}" --k "${k}" --f "${f}" --numseqs "${ns}"
}

duet_cell() {  # duet_cell <base> <label> <B> <K1> <K2> <dfo> <pfo> <ns>
  local base="$1" label="$2" B="$3" K1="$4" K2="$5" dfo="$6" pfo="$7" ns="$8"
  local k=$((K1 + K2)) f=$((dfo + pfo))
  EXTRA_ENV="SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1"
  run_cell "${base}" "${label}" --b "${B}" --numseqs "${ns}" \
    --k "${k}" --f "${f}" --duet --duet_exit_layer 56 \
    --duet_phase1_k "${K1}" --duet_phase2_k "${K2}" \
    --duet_draft_fan_out "${dfo}" --duet_policy b
  EXTRA_ENV=""
}

echo "[$(date -Is)] GPU regime at confirm start:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv

# ---- 0. B=8 K1=1 probe (scan-comparable: ns=12) ----
duet_cell "${OUT}" db8_k1x1_d5p1 8 1 1 5 1 12
PROBE_TPS=$(grep -o "Final Decode Throughput: [0-9.]*" "${OUT}/db8_k1x1_d5p1/run.log" \
  | tail -1 | grep -o "[0-9.]*$" || true)
if awk -v t="${PROBE_TPS:-0}" 'BEGIN{exit !(t > 213.51)}'; then
  B8_K1=1; B8_K2=1; B8_DFO=5; B8_PFO=1; B8_NAME=k1x1_d5p1
else
  B8_K1=2; B8_K2=2; B8_DFO=5; B8_PFO=1; B8_NAME=k2x2_d5p1
fi
echo "[$(date -Is)] === PROBE db8_k1x1_d5p1 tps=${PROBE_TPS:-NONE} vs 213.51 -> B8 confirm shape ${B8_NAME} ==="

# ---- 1. per-B confirms, interleaved D/C ----
for rep in 1 2 3; do
  duet_cell "${CONF}" "b2_duet_r${rep}" 2 6 5 3 1 20
  c_cell    "${CONF}" "b2_c_r${rep}"    2 5 6 20
done

for rep in 1 2 3; do
  duet_cell "${CONF}" "b4_duet_r${rep}" 4 3 3 4 1 20
  c_cell    "${CONF}" "b4_c_r${rep}"    4 3 6 20
done

for rep in 1 2 3; do
  duet_cell "${CONF}" "b8_duet_r${rep}" 8 "${B8_K1}" "${B8_K2}" "${B8_DFO}" "${B8_PFO}" 20
  c_cell    "${CONF}" "b8_c_r${rep}"    8 3 6 20
done

for rep in 1 2 3; do
  duet_cell "${CONF}" "b16_duet_r${rep}" 16 1 1 5 1 16
  c_cell    "${CONF}" "b16_c_r${rep}"    16 2 3 16
done

for rep in 1 2 3; do
  duet_cell "${CONF}" "b32_duet_r${rep}" 32 1 1 4 1 32
  c_cell    "${CONF}" "b32_c_r${rep}"    32 2 2 32
done

echo "[$(date -Is)] GPU regime at confirm end:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
echo "[$(date -Is)] === BSCALE32 CONFIRM DONE ==="
