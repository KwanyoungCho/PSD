#!/usr/bin/env bash
# bscale scan phase — B-scaling gap-filling grid (follow-up to
# pb_sweep/RESULTS.md caveat 3: K1=3 was the B=4 grid edge; B=8 and
# K1=2 unmeasured). Extrapolated optimum trend K1 9 -> 6 -> 3 suggests
# K1 ~ 2-3 at B=8.
#   B=8: 5 shapes + C_b8 anchor (k7 f6)
#   B=4: 3 edge cells (K1=2 pair + k3x3_d5p1)
#   B=1: same-regime anchors E9K24_jit champion + C
# All cells: ns=12 out=256 in=512 seed 42 temp 0.7 --all, jit-short on,
# exit=56, uniform dfo, timeout 1200, one run/cell, GPUs 0-4,
# ports 12970+ ascending.
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/bscale"
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

PORT=12970
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

# ---- B=8 grid (6 cells) ----
duet_cell b8_k2x2_d4p1 8 2 2 4 1
duet_cell b8_k2x2_d5p1 8 2 2 5 1
duet_cell b8_k3x3_d4p1 8 3 3 4 1
duet_cell b8_k3x3_d4p2 8 3 3 4 2
duet_cell b8_k4x4_d3p1 8 4 4 3 1
run_one b8_c --b 8 --k 7 --f 6
PORT=$((PORT + 1))

# ---- B=4 edge cells (3) ----
duet_cell b4_k2x2_d4p1 4 2 2 4 1
duet_cell b4_k2x2_d5p1 4 2 2 5 1
duet_cell b4_k3x3_d5p1 4 3 3 5 1

# ---- B=1 same-regime anchors (2) ----
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one b1_e9k24_jit --b 1 \
    --k 13 --f 3 --duet --duet_exit_layer 56 \
    --duet_phase1_k 9 --duet_phase2_k 4 \
    --duet_draft_fan_out 2 --duet_policy b \
    --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1 )
PORT=$((PORT + 1))
run_one b1_c --b 1 --k 7 --f 6

echo "[$(date -Is)] GPU regime at scan end:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
echo "[$(date -Is)] === BSCALE SCAN DONE ==="
