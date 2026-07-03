#!/usr/bin/env bash
# Gate scan — validate the two DUET-only gates + the deep-narrow regime,
# all on the GPU 2,3,5,6,7 set at PROFILE=0 (docs/duet/09 WS2/WS3).
#
# Cells:
#   A_base     A config vanilla (K1=7 K2=5 dfo=2 pfo=1 exit56) — set baseline
#   C_sd       SD k=7 f=6 — the target to beat, re-measured on this set
#   A_jit      A + SSD_DUET_JIT_SHORT=1 (K2-deep JIT; miss verify width K2+1)
#   A_pod      A + SSD_DUET_PROXY_ON_DRAFT=1 (Policy B off target verify path)
#   A_jit_pod  A + both gates
#   E8_deep16  K1=8 K2=5 phase1 list 2*7,1*2 (sum16 = Marlin-tile-safe) + both
#   E9_deep16  K1=9 K2=5 phase1 list 2*6,1*4 (sum16) + both
#
# Rationale (tax_decomposition): phase1 MQ_LEN 16->18 at K1=8 uniform dfo=2
# crossed the draft Marlin m-tile -> per-forward 2.5->3.6 ms, proxy_wait
# idle vanished, draft overran -> target spec_wait echo +1.9/pos. The
# deep-narrow list keeps MQ_LEN=16 while buying +1/+2 depth.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/gate_scan"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=2,3,5,6,7
export SSD_DIST_PORT=12720
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

DUET_A=(
  --k 12 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 5
  --duet_draft_fan_out 2 --duet_policy b
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

# NOTE: each cell runs in a subshell — bash `VAR=1 func` assignments would
# otherwise PERSIST after the function returns and leak gates across cells.

# --- baselines on this GPU set ---
( export SSD_FORCE_SPLIT_K1K2=1; run_one "A_base" "${DUET_A[@]}" )
( run_one "C_sd" --k 7 --f 6 )

# --- single gates ---
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
  run_one "A_jit" "${DUET_A[@]}" )
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_PROXY_ON_DRAFT=1
  run_one "A_pod" "${DUET_A[@]}" )

# --- both gates ---
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 SSD_DUET_PROXY_ON_DRAFT=1
  run_one "A_jit_pod" "${DUET_A[@]}" )

# --- deep-narrow (Marlin-tile-safe MQ_LEN=16) + both gates ---
( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 SSD_DUET_PROXY_ON_DRAFT=1
  run_one "E8_deep16" \
    --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 8 --duet_phase2_k 5 \
    --duet_draft_fan_out 2 --duet_policy b \
    --duet_split_phase1_fan_out_list 2,2,2,2,2,2,2,1,1 )

( export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 SSD_DUET_PROXY_ON_DRAFT=1
  run_one "E9_deep16" \
    --k 14 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 5 \
    --duet_draft_fan_out 2 --duet_policy b \
    --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1 )

echo ""
echo "=== SUMMARY ==="
for label in A_base C_sd A_jit A_pod A_jit_pod E8_deep16 E9_deep16; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${label}/run.log" 2>/dev/null | tail -1 || echo "CRASH/MISSING")
  echo "  ${label}: ${tps}"
done
echo "[$(date -Is)] === ALL DONE ==="
