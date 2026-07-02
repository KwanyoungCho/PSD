#!/usr/bin/env bash
# Deep-P1 test — convert draft idle (proxy_wait 8.9 ms) into P1 depth.
#
# Thesis: 29% of P1 hits accept the FULL 7-chain (alpha=0.838 fit), so P1
# depth is truncating tokens. Unlike dfo-widening (rejected — substitution
# effect), deepening P1 does not cannibalize P2: it raises P1 hit QUALITY.
# Projection: K1=9 -> L_p1 4.12 (+0.45), TPS ~82-83 even paying +2 verify
# positions and (temporarily) a deeper JIT.
#
# Zero code changes: window budget fits (glue+phase1 = 26.5 < 29 ms proxy
# arrival). Known accepted cost in this scan: jit_K follows K_max, so miss
# steps pay +2 fwds (K1=9) — a real implementation would pin JIT at 7.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/deep_p1_test"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12700
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
  SSD_FORCE_SPLIT_K1K2=1 \
    "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1 || {
      echo "[$(date -Is)] === CRASH ${label} (see run.log) ==="
      return 0
    }
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || echo "NO_TPS")
  echo "[$(date -Is)] === END ${label}: ${tps} ==="
}

# K1=8 (k=13): +1 depth, safest step
run_one "k1_8_k2_5" \
  --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 8 --duet_phase2_k 5 \
  --duet_draft_fan_out 2 --duet_policy b

# K1=9 (k=14): +2 depth, projected TPS 82+
run_one "k1_9_k2_5" \
  --k 14 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 5 \
  --duet_draft_fan_out 2 --duet_policy b

# K1=10 (k=15): window-marginal probe (glue+phase1 ≈ 30 vs proxy 29) — tells
# us where the cliff is and whether glue-removal (SwiftSpec) is the unlock.
run_one "k1_10_k2_5" \
  --k 15 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 10 --duet_phase2_k 5 \
  --duet_draft_fan_out 2 --duet_policy b

echo ""
echo "=== SUMMARY ==="
for label in k1_8_k2_5 k1_9_k2_5 k1_10_k2_5; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${label}/run.log" 2>/dev/null | tail -1 || echo "CRASH/MISSING")
  echo "  ${label}: ${tps}"
done
echo "[$(date -Is)] === ALL DONE ==="
