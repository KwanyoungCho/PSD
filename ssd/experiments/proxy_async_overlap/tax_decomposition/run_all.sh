#!/usr/bin/env bash
# Tax decomposition — attribute the 3.4 ms/pos target-step growth per K1.
#
# deep_p1_test measured (PROFILE=0, outer): target step +3.3-3.4 ms per K1
# position against a projected +1.26 (verify CG slope) — ~2 ms/pos
# unattributed. Draft step grows +4.0/pos with only +2.5 attributed to
# phase1 forwards. Deep-P1 (token-correct, K1=9 tok/step 4.27 > C's 4.15)
# is rejected ONLY because of this tax. If the tax decomposes into
# recoverable systems overhead, K1=8-9 beats C outright.
#
# Design: 3 profile points K1 in {7, 8, 9} (K2=5, exit=56, dfo=2, pfo=1),
# SSD_PROFILE_DUET=1, ns=20 (profiling precedent; ~2.5k steps/cell).
# 3 points distinguish per-label LINEAR growth (true per-position cost)
# from STEP jumps (CG bucket / Marlin tile boundary crossings).
#
# Key questions:
#  Q1 target: how much of +3.4/pos is verify CG compute vs spec_wait
#     (pipeline echo of the draft's +4.0/pos)?
#  Q2 does proxy arrival (t_exit) shift later with K1 (more verify
#     positions before the exit layer) — delaying phase2 start?
#  Q3 draft: where is the +1.5/pos beyond phase1 forwards (glue width,
#     unpack/merge, logits_q response wire)?
#
# NOTE: GPUs 0-1 are held by another user (fnsl1026, vLLM). This and all
# subsequent runs use GPUs 2,3,5,6 (target TP4) + 7 (draft). Slopes are
# internal to this GPU set; the final A-vs-C verdict will re-measure BOTH
# configs on this same set.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/tax_decomposition"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=2,3,5,6,7
export SSD_DIST_PORT=12710
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib

BASE_ARGS=(
  --llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 20
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
  SSD_FORCE_SPLIT_K1K2=1 SSD_PROFILE_DUET=1 SSD_PROFILE_DIR="${outdir}" \
    "${PY}" -O bench/bench.py "${BASE_ARGS[@]}" "$@" \
    > "${outdir}/run.log" 2>&1 || {
      echo "[$(date -Is)] === CRASH ${label} (see run.log) ==="
      return 0
    }
  local tps
  tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || echo "NO_TPS")
  echo "[$(date -Is)] === END ${label}: ${tps} ==="
}

run_one "k1_7" \
  --k 12 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 7 --duet_phase2_k 5 \
  --duet_draft_fan_out 2 --duet_policy b

run_one "k1_8" \
  --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 8 --duet_phase2_k 5 \
  --duet_draft_fan_out 2 --duet_policy b

run_one "k1_9" \
  --k 14 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 5 \
  --duet_draft_fan_out 2 --duet_policy b

echo ""
echo "=== SUMMARY ==="
for label in k1_7 k1_8 k1_9; do
  tps=$(grep "Final Decode Throughput" "${PHASE_DIR}/${label}/run.log" 2>/dev/null | tail -1 || echo "CRASH/MISSING")
  echo "  ${label}: ${tps}"
done
echo "[$(date -Is)] === ALL DONE ==="
