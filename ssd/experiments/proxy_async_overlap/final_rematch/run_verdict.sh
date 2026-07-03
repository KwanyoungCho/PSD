#!/usr/bin/env bash
# FINAL VERDICT — champion DUET config vs C, 5 reps each, interleaved.
# Pre-registered rule (docs/duet/09): DUET mean > C mean AND no overlap of
# mean±2SE bands. Canonical GPU set 0-4.
set -euo pipefail
ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/final_rematch"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12775
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
# Champion = E9K24_jit (K2-sweep winner, 83.34 same-regime vs E9_jit 80.09):
# K1=9 deep-narrow phase1 [2x6,1x4] (sum16, tile-safe), K2=4, exit=56,
# pfo=1, SSD_DUET_JIT_SHORT=1.
CHAMPION_ARGS=(
  --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 4
  --duet_draft_fan_out 2 --duet_policy b
  --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1
)
for rep in 1 2 3 4 5; do
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "V_duet_rep${rep}" "${CHAMPION_ARGS[@]}" )
  ( run_one "V_c_rep${rep}" --k 7 --f 6 )
done
echo ""
echo "=== SUMMARY ==="
for rep in 1 2 3 4 5; do
  for arm in V_duet V_c; do
    tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${arm}_rep${rep}/run.log" 2>/dev/null | tail -1 || echo "CRASH/MISSING")
    echo "  ${arm}_rep${rep}: ${tps}"
  done
done
echo "[$(date -Is)] === VERDICT DONE ==="
