#!/usr/bin/env bash
# bscale32 resume — the original Phase A runner died together with the
# cb16_k3f3 cell crash (2026-07-20 16:21, engine died at 77% generation,
# no traceback; runner shell gone). This script (a) retries cb16_k3f3,
# (b) runs the remaining B=32 C cells, (c) runs Phase B DUET B=16/32
# scan cells. Ports 13260+ (step 2) to avoid any stale sockets.
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/bscale32"
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

PORT=13700
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
  PORT=$((PORT + 2))
  cleanup
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || true)
  echo "[$(date -Is)] === END ${label} rc=${rc}: ${tps:-NO_TPS} ==="
}

# c_cell <B> <k> <f> <ns>
c_cell() {
  local B="$1" k="$2" f="$3" ns="$4"
  run_one "cb${B}_k${k}f${f}" --b "${B}" --k "${k}" --f "${f}" --numseqs "${ns}"
}

# duet_cell <label> <B> <K1> <K2> <dfo> <pfo> <ns>
duet_cell() {
  local label="$1" B="$2" K1="$3" K2="$4" dfo="$5" pfo="$6" ns="$7"
  local k=$((K1 + K2)) f=$((dfo + pfo))
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "${label}" --b "${B}" --numseqs "${ns}" \
      --k "${k}" --f "${f}" --duet --duet_exit_layer 56 \
      --duet_phase1_k "${K1}" --duet_phase2_k "${K2}" \
      --duet_draft_fan_out "${dfo}" --duet_policy b )
  PORT=$((PORT + 2))
}

# balance32 — 균형 조건 기반 B=32 DUET 재탐색 (PROFILE=0, ns=32 out=256).
# 조건①: exit를 당겨 proxy 도착 ≈ P1 종료 (e56 기준 proxy_wait 137ms).
# 조건②: 당긴 exit로 생긴 draft 잔여 시간을 pfo(=P2 budget)/dfo로 재배분.
# 기존 e56 베이스라인: k1x1_d4p1 290.59, k2x2_d4p1 270.82; C-opt 304.0.
duet_cell_e() {
  local label="$1" B="$2" K1="$3" K2="$4" dfo="$5" pfo="$6" ns="$7" exit="$8"
  local k=$((K1 + K2)) f=$((dfo + pfo))
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "${label}" --b "${B}" --numseqs "${ns}" \
      --k "${k}" --f "${f}" --duet --duet_exit_layer "${exit}" \
      --duet_phase1_k "${K1}" --duet_phase2_k "${K2}" \
      --duet_draft_fan_out "${dfo}" --duet_policy b )
  PORT=$((PORT + 2))
}
echo "[$(date -Is)] === BALANCE32 GROUP A: exit sweep ==="
duet_cell_e bal_k1x1_d4p1_e48 32 1 1 4 1 32 48
duet_cell_e bal_k1x1_d4p1_e40 32 1 1 4 1 32 40
duet_cell_e bal_k1x1_d4p1_e32 32 1 1 4 1 32 32
duet_cell_e bal_k2x2_d4p1_e40 32 2 2 4 1 32 40
duet_cell_e bal_k2x2_d4p1_e32 32 2 2 4 1 32 32
echo "[$(date -Is)] === BALANCE32 GROUP B: rebalance (pfo/dfo) ==="
duet_cell_e bal_k1x1_d4p2_e40 32 1 1 4 2 32 40
duet_cell_e bal_k1x1_d4p3_e40 32 1 1 4 3 32 40
duet_cell_e bal_k1x1_d6p1_e40 32 1 1 6 1 32 40
duet_cell_e bal_k1x1_d6p2_e40 32 1 1 6 2 32 40
duet_cell_e bal_k2x2_d4p2_e32 32 2 2 4 2 32 32
echo "[$(date -Is)] === BALANCE32 SCAN DONE ==="
