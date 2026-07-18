#!/usr/bin/env bash
# M6 fix — re-run the M5 DUET cells only (B in {1,2,4}), identical args to
# m5_sweep/run_all.sh (champion E9K24_jit, ns=20 out=256 seed 42, GPUs 0-4),
# ports 12911-12913. C cells are NOT re-run (no C-side code touched).
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/m6_fix"
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
DUET_ARGS=(
  --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 4
  --duet_draft_fan_out 2 --duet_policy b
  --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1
)

cleanup() {
  pkill -9 -u chokwans99 -f "bench/bench.py" 2>/dev/null || true
  pkill -9 -u chokwans99 -f "multiprocessing.spawn import spawn_main" 2>/dev/null || true
  pkill -9 -u chokwans99 -f "multiprocessing.resource_tracker" 2>/dev/null || true
  sleep 6
}

PORT=12911
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

echo "[$(date -Is)] GPU regime at start:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv

for B in 1 2 4; do
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "duet_b${B}" --b "${B}" "${DUET_ARGS[@]}" )
  PORT=$((PORT + 1))
done

echo "[$(date -Is)] GPU regime at end:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
echo "[$(date -Is)] === M6 DUET CELLS DONE ==="
