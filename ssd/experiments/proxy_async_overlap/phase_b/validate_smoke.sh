#!/usr/bin/env bash
# Phase B validation smoke test.
#
# Greedy temp=0 with 8B model — quick byte-identical check between
# SSD_PROFILE_MESA=0 and =1 to verify Phase B (aligned trace) introduces
# no behavioral change.

set -euo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/phase_b"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"

cd "${ROOT}"
source ${ROOT}/env.sh

mkdir -p "${PHASE_DIR}/off" "${PHASE_DIR}/on"

# Common args: small 8B run, MESA split-K1/K2 mode, greedy
COMMON_ARGS=(
  --llama --size 8
  --gpus 3 --b 1 --temp 0 --seed 42 --numseqs 8
  --input_len 128 --output_len 128 --max_model_len 2048
  --async --spec --k 5 --f 3
  --mesa --mesa_exit_layer 21 --mesa_phase1_k 3 --mesa_phase2_k 2
  --mesa_draft_fan_out 2 --mesa_policy b
)

run_one() {
  local label="$1"
  local profile="$2"
  local outdir="${PHASE_DIR}/${label}"

  echo "[$(date -Is)] === START ${label} (PROFILE_MESA=${profile}) ==="
  pkill -9 -f "bench.py" 2>/dev/null || true
  sleep 3

  SSD_PROFILE_DIR="${outdir}" SSD_PROFILE_MESA="${profile}" SSD_FORCE_SPLIT_K1K2=1 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
    > "${outdir}/run.log" 2>&1

  echo "[$(date -Is)] === END ${label} ==="
}

# OFF: PROFILE_MESA=0 (baseline, no anchor/context cost)
run_one "off" 0

# ON:  PROFILE_MESA=1 (anchor + context + new schema)
run_one "on" 1

echo "[$(date -Is)] === Phase B smoke complete ==="
echo "Compare:"
echo "  diff <(grep -E 'Generation:' ${PHASE_DIR}/off/run.log) <(grep -E 'Generation:' ${PHASE_DIR}/on/run.log)"
echo "JSON:"
echo "  ls ${PHASE_DIR}/on/mesa_profile_*.json"
