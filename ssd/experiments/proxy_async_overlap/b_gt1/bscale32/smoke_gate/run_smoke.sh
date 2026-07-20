#!/usr/bin/env bash
# bscale32 gate smoke — first-ever B=16 / B=32 DUET runs after the
# config cap lift 8 -> 32. Purpose: (a) engine survives (CG buckets
# {16,32} capture, no config assert), (b) KV pool holds ns=B seqs at
# in=512 out=128 without preemption collapse, (c) sane decode TPS.
# Short cells: out=128, ns=B (one full wave), one run/cell.
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/bscale32/smoke_gate"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_PROFILE_DUET=0

base_args() {
  local ns="$1"
  echo --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --quant_awq \
    --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
    --quant_group_size 128 \
    --gpus 5 --temp 0.7 --seed 42 --numseqs "${ns}" \
    --input_len 512 --output_len 128 --all --max_model_len 2048 \
    --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
    --quant_awq_draft \
    --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
    --async --spec
}

cleanup() {
  pkill -9 -u chokwans99 -f "bench/bench.py" 2>/dev/null || true
  pkill -9 -u chokwans99 -f "multiprocessing.spawn import spawn_main" 2>/dev/null || true
  pkill -9 -u chokwans99 -f "multiprocessing.resource_tracker" 2>/dev/null || true
  sleep 6
}

PORT=13100
run_one() {
  local label="$1" ns="$2"; shift 2
  local outdir="${OUT}/${label}"
  mkdir -p "${outdir}"
  cleanup
  echo "[$(date -Is)] === START ${label} (port ${PORT}) ==="
  SSD_DIST_PORT="${PORT}" timeout -k 30 1500 \
    "${PY}" -O bench/bench.py $(base_args "${ns}") "$@" \
    > "${outdir}/run.log" 2>&1
  local rc=$?
  PORT=$((PORT + 2))
  cleanup
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || true)
  echo "[$(date -Is)] === END ${label} rc=${rc}: ${tps:-NO_TPS} ==="
}

duet_cell() {
  local label="$1" B="$2" K1="$3" K2="$4" dfo="$5" pfo="$6"
  local k=$((K1 + K2)) f=$((dfo + pfo))
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "${label}" "${B}" --b "${B}" \
      --k "${k}" --f "${f}" --duet --duet_exit_layer 56 \
      --duet_phase1_k "${K1}" --duet_phase2_k "${K2}" \
      --duet_draft_fan_out "${dfo}" --duet_policy b )
  PORT=$((PORT + 2))
}

duet_cell smoke_b16_k2x2_d5p1 16 2 2 5 1
duet_cell smoke_b32_k2x2_d5p1 32 2 2 5 1
run_one smoke_b32_c 32 --b 32 --k 7 --f 6
echo "[$(date -Is)] === BSCALE32 SMOKE DONE ==="
