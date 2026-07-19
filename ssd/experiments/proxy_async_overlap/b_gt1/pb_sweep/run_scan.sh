#!/usr/bin/env bash
# pb_sweep scan phase — per-B DUET shape sweep (docs/duet/13 follow-up to
# verdict/RESULTS.md §4.3: "are K1/K2/dfo/pfo actually optimal per B?").
# Grid (constraint K2<=K1, k=K1+K2, f=dfo+pfo; guided by: verify rows
# ~ K1+1 dominates step time; K1-K2 gap small neutralizes vk_max padding;
# draft idle 34.5 ms at B=4 can fund dfo/pfo):
#   B=4: fat5 rerun anchor + 7 neighbors + C_b4 anchor rerun (k7 f6)
#   B=2: fat7 + 4 neighbors + C_b2 anchor rerun
# All cells: ns=12 out=256 in=512 seed 42 temp 0.7 --all, jit-short on,
# exit=56, timeout 1200, one run/cell, GPUs 0-4, ports 12930+ ascending.
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/pb_sweep"
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

PORT=12930
run_one() {
  local label="$1"; shift
  local outdir="${OUT}/${label}"
  mkdir -p "${outdir}"
  cleanup
  echo "[$(date -Is)] === START ${label} (port ${PORT}) ==="
  SSD_DIST_PORT="${PORT}" timeout -k 30 1200 \
    "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1
  local rc=$?
  PORT=$((PORT + 1))
  cleanup
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || true)
  echo "[$(date -Is)] === END ${label} rc=${rc}: ${tps:-NO_TPS} ==="
}

# duet_cell <label> <B> <K1> <K2> <dfo> <pfo>
duet_cell() {
  local label="$1" B="$2" K1="$3" K2="$4" dfo="$5" pfo="$6"
  local k=$((K1 + K2)) f=$((dfo + pfo))
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "${label}" --b "${B}" \
      --k "${k}" --f "${f}" --duet --duet_exit_layer 56 \
      --duet_phase1_k "${K1}" --duet_phase2_k "${K2}" \
      --duet_draft_fan_out "${dfo}" --duet_policy b )
  PORT=$((PORT + 1))
}

echo "[$(date -Is)] GPU regime at scan start:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv

# ---- B=4 grid (9 cells) ----
duet_cell b4_k5x4_d3p1 4 5 4 3 1    # fat5 rerun anchor
duet_cell b4_k4x3_d3p1 4 4 3 3 1
duet_cell b4_k4x4_d3p1 4 4 4 3 1
duet_cell b4_k5x4_d4p1 4 5 4 4 1
duet_cell b4_k5x4_d3p2 4 5 4 3 2
duet_cell b4_k6x5_d3p1 4 6 5 3 1
duet_cell b4_k5x5_d3p1 4 5 5 3 1
duet_cell b4_k3x3_d4p1 4 3 3 4 1
run_one b4_c --b 4 --k 7 --f 6      # C_b4 anchor rerun
PORT=$((PORT + 1))

# ---- B=2 grid (6 cells) ----
duet_cell b2_k7x4_d2p1 2 7 4 2 1    # fat7
duet_cell b2_k7x6_d2p1 2 7 6 2 1
duet_cell b2_k6x5_d2p1 2 6 5 2 1
duet_cell b2_k5x4_d3p1 2 5 4 3 1
duet_cell b2_k6x5_d3p1 2 6 5 3 1
run_one b2_c --b 2 --k 7 --f 6      # C_b2 anchor rerun

echo "[$(date -Is)] GPU regime at scan end:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
echo "[$(date -Is)] === PB_SWEEP SCAN DONE ==="
