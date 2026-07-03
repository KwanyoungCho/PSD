#!/usr/bin/env bash
# K2 shrink probes under jit-short economics (docs/duet/09 WS3d).
# Three nulls (pod/topm/replica) proved target-busy no longer binds at E9;
# draft busy (~47.5ms) is co-critical. K2 shrink cuts the draft chain
# (phase2 forwards), the miss/short verify width (K2+1), and JIT depth
# at once. Token cost: L_p2 1.99 -> ~1.85 (K2=4) / 1.42 (K2=3).
set -euo pipefail
ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/final_rematch"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12770
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
  pkill -9 -u chokwans99 -f "python -O bench/bench.py" 2>/dev/null || true
  sleep 5
  "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1 || {
      echo "[$(date -Is)] === CRASH ${label} ==="
      return 0
    }
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || echo "NO_TPS")
  echo "[$(date -Is)] === END ${label}: ${tps} ==="
}
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one "E9K24_jit" \
    --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 4 \
    --duet_draft_fan_out 2 --duet_policy b \
    --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1 )
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one "E9K23_jit" \
    --k 12 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 3 \
    --duet_draft_fan_out 2 --duet_policy b \
    --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1 )
echo "[$(date -Is)] === K2SWEEP DONE ==="
