#!/usr/bin/env bash
# pb_sweep fixup — reruns the two C anchor cells that failed in the first
# scan pass (a stray positional arg in the original run_scan.sh call, rc=2
# before model load; fixed in run_scan.sh) + one extra B=4 scan cell
# k3x3_d4p2 (winner shape + pfo=2, the two top scan effects combined).
# Same regime as the scan: ns=12 out=256 in=512 seed 42 temp 0.7, GPUs 0-4.
# Ports 12946-12948.
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

PORT=12946
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

run_one b4_c --b 4 --k 7 --f 6
run_one b2_c --b 2 --k 7 --f 6
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one b4_k3x3_d4p2 --b 4 --k 6 --f 6 --duet --duet_exit_layer 56 \
    --duet_phase1_k 3 --duet_phase2_k 3 --duet_draft_fan_out 4 --duet_policy b )

echo "[$(date -Is)] === PB_SWEEP FIXUP DONE ==="
