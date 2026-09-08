#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 LABEL chain|tree TEMP SEED [OUTPUT=512] [DATA]" >&2
  exit 2
fi

LABEL=$1
POLICY=$2
TEMP=$3
SEED=$4
OUTPUT=${5:-512}
ROOT=/home/eslab/chokwans99/PSD/ssd/experiments/proxy_async_overlap/tree_sweep/phase_difficulty_k8_exit56_20260814
DATA=${6:-${ROOT}/balanced_120q.jsonl}
PY=/home/eslab/chokwans99/baseline/.venv-ssd/bin/python
RUNNER=/home/eslab/chokwans99/baseline/runners/run_duet.py
OUT=${ROOT}/runs/${LABEL}
mkdir -p "${OUT}"

# Optional tree-shape overrides for the AL-only K1=K2 calibration.  Defaults
# reproduce the original matched 12/12 diagnostic exactly.
N1=${N1:-12}
M1=${M1:-12}
N2=${N2:-12}
M2=${M2:-12}
P1_START_THRESHOLD=${P1_START_THRESHOLD:-0}
P1_CONF_THRESHOLD=${P1_CONF_THRESHOLD:-0}
P2_PROXY_THRESHOLD=${P2_PROXY_THRESHOLD:-0.01}
P2_CONF_THRESHOLD=${P2_CONF_THRESHOLD:-0.01}
C_TENSOR=${C_TENSOR:-2}

case "${POLICY}" in
  chain) P1_TREE=off; P2_TREE=off ;;
  tree)  P1_TREE=on;  P2_TREE=on ;;
  *) echo "unknown policy: ${POLICY}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
export SSD_SEED=${SEED}
export SSD_PROFILE_DUET=0
export SSD_TREE_TRACE=0
export SSD_TREE_CALIBRATION=0
# K1=K2=8 with matched 12-node tree buckets needs larger FlashInfer planning
# arenas than the production K2=5 shape.  This changes capture memory only,
# not tree construction, verification, or measured AL.
export SSD_TREE_VERIFY_WORKSPACE_MB=${SSD_TREE_VERIFY_WORKSPACE_MB:-224}
export SSD_TREE_EXEC_WORKSPACE_MB=${SSD_TREE_EXEC_WORKSPACE_MB:-112}
export SSD_P1_TREE_EXEC_WORKSPACE_MB=${SSD_P1_TREE_EXEC_WORKSPACE_MB:-112}
if [[ -n "${NODE_AUDIT:-}" ]]; then
  export SSD_TREE_NODE_AUDIT="${OUT}/node_audit"
fi

EXTRA_ARGS=()
if [[ -n "${RUN_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${RUN_LIMIT}")
fi

timeout --signal=TERM --kill-after=60s 300m \
  "${PY}" -O "${RUNNER}" \
  --target facebook/layerskip-llama2-70B \
  --draft TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --gpus 3 \
  --k1 8 --k2 8 --exit-layer 56 \
  --p1-fanout 3 --p2-budget 15 --proxy-top-k 28 \
  --temp "${TEMP}" --top_p 1.0 \
  --max_new_tokens "${OUTPUT}" --max_model_len 2048 \
  --template raw --warmup 2 --seed "${SEED}" \
  --p1-tree "${P1_TREE}" --p2-tree "${P2_TREE}" \
  --p1-allocation-policy backbone \
  --roots-per-position 3 --root-count 10 --c-tensor "${C_TENSOR}" \
  --n1 "${N1}" --p1-verify-nodes "${M1}" \
  --n2 "${N2}" --p2-verify-nodes "${M2}" \
  --p1-start-threshold "${P1_START_THRESHOLD}" \
  --p1-conf-threshold "${P1_CONF_THRESHOLD}" \
  --p2-proxy-threshold "${P2_PROXY_THRESHOLD}" \
  --p2-conf-threshold "${P2_CONF_THRESHOLD}" \
  --emit-phase-trace --resume \
  "${EXTRA_ARGS[@]}" \
  --data "${DATA}" --out "${OUT}/raw.jsonl" \
  >"${OUT}/run.log" 2>&1
