#!/usr/bin/env bash
# bscale32 Phase A — C (plain async-SD) per-B optimization scan.
# Fairness fix: prior campaigns fixed C at its B=1-optimal k7f6 at every B
# while DUET got per-B shape retuning. This scan per-B-optimizes C.
# Grid:
#   B=2,4,8 (ns=12): k7f6 (fresh anchor), k5f6, k3f6, k5f3, k3f3
#   B=16 (ns=16):    k7f6, k5f6, k3f6, k5f3, k3f3
#   B=32 (ns=32):    k3f6, k5f6 (OOM boundary probe), k5f3, k3f3, k2f3, k2f2
#     (k7f6 @ B=32 = known DNF from gate smoke — not rerun)
# All cells: out=256 in=512 seed 42 temp 0.7 --all, exit-irrelevant (no DUET),
# timeout 1800, one run/cell, GPUs 0-4, ports 13200+ (step 2).
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

PORT=13200
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

echo "[$(date -Is)] GPU regime at scan start:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv

# ---- B=2 (ns=12) ----
c_cell 2 7 6 12
c_cell 2 5 6 12
c_cell 2 3 6 12
c_cell 2 5 3 12
c_cell 2 3 3 12

# ---- B=4 (ns=12) ----
c_cell 4 7 6 12
c_cell 4 5 6 12
c_cell 4 3 6 12
c_cell 4 5 3 12
c_cell 4 3 3 12

# ---- B=8 (ns=12) ----
c_cell 8 7 6 12
c_cell 8 5 6 12
c_cell 8 3 6 12
c_cell 8 5 3 12
c_cell 8 3 3 12

# ---- B=16 (ns=16) ----
c_cell 16 7 6 16
c_cell 16 5 6 16
c_cell 16 3 6 16
c_cell 16 5 3 16
c_cell 16 3 3 16

# ---- B=32 (ns=32) ---- k7f6 known DNF (gate smoke), not rerun
c_cell 32 3 6 32
c_cell 32 5 6 32   # boundary probe: 1152 verify rows, likely OOM — DNF is data
c_cell 32 5 3 32
c_cell 32 3 3 32
c_cell 32 2 3 32
c_cell 32 2 2 32

echo "[$(date -Is)] GPU regime at scan end:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
echo "[$(date -Is)] === BSCALE32 C SCAN DONE ==="
