#!/usr/bin/env bash
# Verdict Exp2 — B>1-shape retune probes at B=4 (fewer, fatter draft forwards).
#   fat7: K1=7 K2=4 (k=11), uniform dfo=2 (no fan_out_list -> [2]*8 = 16 rows
#         phase1/seq), 11 draft forwards. f=3 -> pfo=1, phase2 budget 8.
#   fat5: K1=5 K2=4 (k=9), uniform dfo=3 ([3]*6 = 18 rows phase1/seq),
#         9 draft forwards. Needs f=4 (config: dfo < async_fan_out) -> pfo=1,
#         phase2 budget 6.
# Same base args as m6_fix duet cells (ns=20 out=256 seed 42, GPUs 0-4),
# SSD_PROFILE_DUET=0, jit-short on. Ports 12921-12922.
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
OUT="${ROOT}/experiments/proxy_async_overlap/b_gt1/verdict"
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
  --gpus 5 --b 4 --temp 0.7 --seed 42 --numseqs 20
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

run_one() {
  local label="$1" port="$2"; shift 2
  local outdir="${OUT}/${label}"
  mkdir -p "${outdir}"
  cleanup
  echo "[$(date -Is)] === START ${label} (port ${port}) ==="
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    SSD_DIST_PORT="${port}" timeout -k 30 1800 \
      "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
      > "${outdir}/run.log" 2>&1 )
  local rc=$?
  cleanup
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || true)
  echo "[$(date -Is)] === END ${label} rc=${rc}: ${tps:-NO_TPS} ==="
}

run_one fat7 12921 \
  --k 11 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 4 \
  --duet_draft_fan_out 2 --duet_policy b

run_one fat5 12922 \
  --k 9 --f 4 --duet --duet_exit_layer 56 --duet_phase1_k 5 --duet_phase2_k 4 \
  --duet_draft_fan_out 3 --duet_policy b

echo "[$(date -Is)] === RETUNE CELLS DONE ==="
