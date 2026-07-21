#!/usr/bin/env bash
# bscale32 Phase B — DUET B=16/32 scan (8 cells).
# Shape law so far: optimal K1 9 -> 6 -> 3 -> 2 for B=1 -> 2 -> 4 -> 8 (K2=K1);
# expect K1 in {1,2} at B=16/32. K1=1 has NEVER been run — may crash; if so
# the failure is recorded and the campaign continues.
#   B=16 (ns=16): k2x2_d5p1, k2x2_d4p1, k3x3_d4p1, k1x1_d5p1
#   B=32 (ns=32): k2x2_d5p1, k2x2_d4p1, k1x1_d5p1, k1x1_d7p1
# All cells: out=256 in=512 seed 42 temp 0.7 --all, jit-short on, exit=56,
# timeout 1800, one run/cell, GPUs 0-4, ports 13260+ (step 2).
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

PORT=13260
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

echo "[$(date -Is)] GPU regime at scan start:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv

# ---- B=16 (ns=16) ----
duet_cell b16_k2x2_d5p1 16 2 2 5 1 16
duet_cell b16_k2x2_d4p1 16 2 2 4 1 16
duet_cell b16_k3x3_d4p1 16 3 3 4 1 16
duet_cell b16_k1x1_d5p1 16 1 1 5 1 16

# ---- B=32 (ns=32) ----
duet_cell b32_k2x2_d5p1 32 2 2 5 1 32
duet_cell b32_k2x2_d4p1 32 2 2 4 1 32
duet_cell b32_k1x1_d5p1 32 1 1 5 1 32
duet_cell b32_k1x1_d7p1 32 1 1 7 1 32

echo "[$(date -Is)] GPU regime at scan end:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
echo "[$(date -Is)] === BSCALE32 DUET SCAN DONE ==="
