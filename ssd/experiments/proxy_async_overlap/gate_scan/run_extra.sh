#!/usr/bin/env bash
# Disambiguation cells after the main scan: is the jit×pod interaction real,
# and does deep-narrow prefer jit-only?
set -euo pipefail
ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/gate_scan"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=2,3,5,6,7
export SSD_DIST_PORT=12721
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
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 50
  --input_len 512 --output_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --async --spec
)
run_one() {
  local label="$1"; shift
  local outdir="${PHASE_DIR}/${label}"
  mkdir -p "${outdir}"
  echo "[$(date -Is)] === START ${label} ==="
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 5
  "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1 || {
      echo "[$(date -Is)] === CRASH ${label} (see run.log) ==="
      return 0
    }
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || echo "NO_TPS")
  echo "[$(date -Is)] === END ${label}: ${tps} ==="
}
E9_ARGS=(
  --k 14 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 5
  --duet_draft_fan_out 2 --duet_policy b
  --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1
)
# E9 with jit only (no pod) — deep-narrow's preferred gate set?
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one "E9_jit" "${E9_ARGS[@]}" )
# repeat combo — noise vs real interaction
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 SSD_DUET_PROXY_ON_DRAFT=1
  run_one "A_jit_pod_r2" --k 12 --f 3 --duet --duet_exit_layer 56 \
    --duet_phase1_k 7 --duet_phase2_k 5 --duet_draft_fan_out 2 --duet_policy b )
# repeat jit — anchor its band
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one "A_jit_r2" --k 12 --f 3 --duet --duet_exit_layer 56 \
    --duet_phase1_k 7 --duet_phase2_k 5 --duet_draft_fan_out 2 --duet_policy b )
echo "[$(date -Is)] === EXTRA DONE ==="
