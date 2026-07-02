#!/usr/bin/env bash
# Best-config rematch — A (DUET paper best), C (async SD best), D (async SD
# f-matched) × 3 reps each, PROFILE_DUET=0 cold path (paper-headline setting).
#
# Prior numbers being re-validated (pre-Batch-1/2 code):
#   A: DUET K1=7 K2=5 exit=56 dfo=2 pfo=1 (k=12 f=3)  → 80.42 tok/s
#   C: async SD k=7 f=6                                → 83.65 tok/s
#   D: async SD k=7 f=3                                → 80.35 tok/s
#
# 3 reps per config, identical command per rep (seed fixed; run-to-run
# variance comes from async timing → cache-path divergence at temp=0.7).
# Report mean ± CoV; two configs differ only if bands don't overlap.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/best_config_rematch"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

# GPU 0-3 = TP target ranks (filled sequentially from 0), GPU 4 = draft (last).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12690
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
  local rep="$1"; shift
  local outdir="${PHASE_DIR}/${label}/rep${rep}"
  mkdir -p "${outdir}"

  echo "[$(date -Is)] === START ${label} rep${rep} ==="
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 5

  # SSD_FORCE_SPLIT_K1K2 only matters for DUET runs; harmless for SD.
  SSD_FORCE_SPLIT_K1K2=1 \
    "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1

  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || echo "NO_TPS")
  echo "[$(date -Is)] === END ${label} rep${rep}: ${tps} ==="
}

A_ARGS=(--k 12 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 5 --duet_draft_fan_out 2 --duet_policy b)
C_ARGS=(--k 7 --f 6)
D_ARGS=(--k 7 --f 3)

# Interleave configs across reps so slow drift (thermals, host load) spreads
# evenly instead of biasing one config.
for rep in 1 2 3; do
  run_one "A_duet_k7k5"  "${rep}" "${A_ARGS[@]}"
  run_one "C_sd_k7f6"    "${rep}" "${C_ARGS[@]}"
  run_one "D_sd_k7f3"    "${rep}" "${D_ARGS[@]}"
done

echo ""
echo "=== SUMMARY ==="
for label in A_duet_k7k5 C_sd_k7f6 D_sd_k7f3; do
  echo "--- ${label} ---"
  for rep in 1 2 3; do
    grep -H "Final Decode Throughput" "${PHASE_DIR}/${label}/rep${rep}/run.log" | tail -1 || true
  done
done
echo "[$(date -Is)] === ALL DONE ==="
