#!/usr/bin/env bash
# WS4 final rematch — E9_jit (DUET champion) vs C (SD best), 3 reps each,
# interleaved per cycle to spread drift. PROFILE=0.
# Pre-registered rule (docs/duet/09): DUET mean > C mean AND no band overlap.
#
# CANONICAL GPU SET 0-4 (target TP4 on 0-3, draft on 4) — the neighbor's
# vLLM on GPUs 0-1 exited at ~11:20, so the set every prior series
# (A 81.24, C 82.72±0.41) was measured on is available again. The first
# 1.5 reps run on 2,3,5,6,7 are archived in altset_partial/ (E9jit_rep1
# 80.11, C_rep1 killed mid-run at the switch).
set -euo pipefail
ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/final_rematch"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12735
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
for rep in 1 2 3; do
  ( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
    run_one "E9jit_rep${rep}" "${E9_ARGS[@]}" )
  ( run_one "C_rep${rep}" --k 7 --f 6 )
done
echo ""
echo "=== SUMMARY ==="
for label in E9jit_rep1 C_rep1 E9jit_rep2 C_rep2 E9jit_rep3 C_rep3; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${label}/run.log" 2>/dev/null | tail -1 || echo "CRASH/MISSING")
  echo "  ${label}: ${tps}"
done
echo "[$(date -Is)] === REMATCH DONE ==="
