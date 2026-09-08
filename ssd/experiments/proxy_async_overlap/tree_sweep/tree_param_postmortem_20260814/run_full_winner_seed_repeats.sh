#!/usr/bin/env bash
# Repeat the REPORT.md optimized DUET-tree configuration on two independent
# sampler seeds, then build the same 2,048-safe subset used by the paper
# best-subtask composite.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=/home/eslab/chokwans99/baseline
PAPER=/home/eslab/chokwans99/DUET_PAPER_RESULTS
PY=${BASE}/.venv-ssd/bin/python
FILTER=${PAPER}/scripts/metrics/filter_context_safe_subset.py
METRICS=${BASE}/analysis/question_level_metrics.py

run_seed() {
  local seed=$1
  local port=$2
  local tag="winner_k8_k5_e49_n10m10_s${seed}"
  local arm="${HERE}/full/${tag}"

  GPU_SET=5,6,7 PORT=${port} \
    K1=8 K2=5 EXIT_LAYER=49 \
    N1=14 M1=12 N2=10 M2=10 \
    RPP=3 ROOT_COUNT=10 C_TENSOR=2 \
    P1_START=0 P1_CONF=0 P2_PROXY=0.01 P2_CONF=0.01 \
    OUTLEN=1024 SEED=${seed} \
    bash "${HERE}/run_full_arm.sh" "${tag}"

  "${PY}" "${FILTER}" "${arm}/raw.jsonl" \
    "${arm}/raw_ctx2048_safe.jsonl" --context-limit 2048 \
    >"${arm}/ctx2048_safe_filter.json"
  "${PY}" "${METRICS}" "${arm}/raw_ctx2048_safe.jsonl" --json \
    >"${arm}/metrics_ctx2048_safe.json"
}

run_seed 1 19151
run_seed 123 19152

# Also materialize the common-subset view of the existing seed-42 winner.
seed42_arm=${HERE}/full/winner_k8_k5_e49_n10m10
"${PY}" "${FILTER}" "${seed42_arm}/raw.jsonl" \
  "${seed42_arm}/raw_ctx2048_safe.jsonl" --context-limit 2048 \
  >"${seed42_arm}/ctx2048_safe_filter.json"
"${PY}" "${METRICS}" "${seed42_arm}/raw_ctx2048_safe.jsonl" --json \
  >"${seed42_arm}/metrics_ctx2048_safe.json"

echo "[$(date -Is)] optimized tree seed repeats complete"
