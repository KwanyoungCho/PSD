#!/usr/bin/env bash
# DUET window-budget widening sweep — the cells that were unreachable before
# the B=0 guard (2041ac9) + split-aware lookahead (ca27aa7) fixes.
#
# Rationale (best_config_rematch/RESULTS.md): DUET's deficit vs SD's best
# (k=7 f=6) is pure tree width. Draft windows measured on the A config:
#   Window 1 (glue+phase1 before proxy arrival): 22.4 ms used / ~29 ms budget
#   Window 2 (phase2+merge after proxy):          12.5 ms used / ~17.5 ms budget
# dfo raises Window-1 use sub-linearly (memory-bound draft); pfo raises
# Window-2 use. Cells below are chosen to stay inside both budgets.
#
# Scan pass: 1 rep per cell. The best cell then gets +2 reps (separate script
# or manual) before any claim vs C (82.72 ± 0.41).

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/duet_wide_sweep"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12695
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

# Cell 1: (dfo=3, pfo=1) f=4 — safe step, Window 1 +~0.7 ms
run_one "dfo3_pfo1_f4" \
  --k 12 --f 4 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 5 \
  --duet_draft_fan_out 3 --duet_policy b

# Cell 2: (dfo=4, pfo=1) f=5 — Window 1 +~1.4 ms (batch 16→32, sub-linear)
run_one "dfo4_pfo1_f5" \
  --k 12 --f 5 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 5 \
  --duet_draft_fan_out 4 --duet_policy b

# Cell 3: (dfo=4, pfo=2) f=6 — same f as SD best; Window 2 +~2 ms
run_one "dfo4_pfo2_f6" \
  --k 12 --f 6 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 5 \
  --duet_draft_fan_out 4 --duet_policy b

# Cell 4: K2=3 variant at dfo=4 — Window 2 relief (phase2 5→3 fwds, −4 ms);
# L_p2≈2.0 means depth 3 loses almost nothing. k = K1+K2 = 10.
run_one "dfo4_pfo1_f5_k2_3" \
  --k 10 --f 5 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 3 \
  --duet_draft_fan_out 4 --duet_policy b

echo ""
echo "=== SUMMARY ==="
for label in dfo3_pfo1_f4 dfo4_pfo1_f5 dfo4_pfo2_f6 dfo4_pfo1_f5_k2_3; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${label}/run.log" 2>/dev/null | tail -1 || echo "CRASH/MISSING")
  echo "  ${label}: ${tps}"
done
echo "[$(date -Is)] === ALL DONE ==="
