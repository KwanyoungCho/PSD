#!/usr/bin/env bash
# Full Spec-Bench chain control matched to the selected DUET P1+P2-tree arm.
# Only the P1/P2 candidate topology is changed from tree to chain.
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
DATA=${BASE}/data/specbench_full.jsonl
OUT=${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_matched_chain_seed42_20260813
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU_SET=${GPU_SET:-5,6,7}
SEED=${SEED:-42}
OUTLEN=1024
TIMEOUT_MIN=${TIMEOUT_MIN:-240}
PORT=${PORT:-18720}
STEM=duet_chain_matched_treecfg_s${SEED}_o${OUTLEN}_ctx2048
RAW=${OUT}/${STEM}.jsonl
LOG=${OUT}/${STEM}.log

mkdir -p "${OUT}"
cd "${BASE}"

export DUET_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export HF_HOME=/home/eslab/models
export SSD_HF_CACHE=/home/eslab/models/hub
export SSD_DATASET_DIR=${BASE}/data
export TORCH_CUDA_ARCH_LIST=12.0
export SSD_CUDA_ARCH=12.0
export SSD_ATTN_BACKEND=auto
export SSD_CHAIN_PROXY_GRAPH=1
export SSD_DUET_EXIT_REPLICA=1
export SSD_ASYNC_PROXY_SEND=1
export SSD_PROXY_STREAM=0

# The selected tree arm sets these to one.  They are disabled here because
# both topology policies are off and no tree-only allocation/execution work
# should contaminate the matched chain control.
export SSD_TREE_EXEC=0
export SSD_TREE_ARENA=0
export SSD_TREE_PROXY_GRAPH=0
export SSD_TREE_EXEC_WARMUP=0
export SSD_TREE_TOPOLOGY_GPU=0
export SSD_P1_RERANK_PRECOMPUTE=0
export SSD_PROFILE=0
export SSD_PROFILE_DUET=0
export SSD_PROFILE_DUET_DETAIL=0

if [[ $(wc -l < "${DATA}") -ne 560 ]]; then
  echo "expected 560-turn full Spec-Bench input: ${DATA}" >&2
  exit 2
fi

rows=0
[[ -e "${RAW}" ]] && rows=$(wc -l < "${RAW}")
if [[ ${rows} -eq 560 ]]; then
  echo "[$(date -Is)] SKIP matched chain complete: ${RAW}"
  exit 0
fi
if [[ ${rows} -gt 560 ]]; then
  echo "invalid row count ${rows}: ${RAW}" >&2
  exit 3
fi

echo "[$(date -Is)] START matched chain seed=${SEED} resume_rows=${rows}"
SSD_DIST_PORT="${PORT}" \
  timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
  "${PY}" -O "${RUNNER}" \
    --target "${TARGET}" --draft "${DRAFT}" --gpus 3 \
    --k1 8 --k2 4 --exit-layer 56 \
    --p1-fanout 3 --p2-budget 15 --proxy-top-k 28 \
    --temp 0.7 --top_p 1.0 --max_new_tokens "${OUTLEN}" \
    --max_model_len 2048 \
    --template raw --warmup 2 \
    --p1-tree off --p2-tree off --p1-allocation-policy backbone \
    --roots-per-position 3 --root-count 10 --c-tensor 2 \
    --n1 14 --p1-verify-nodes 12 --n2 8 --p2-verify-nodes 8 \
    --p1-start-threshold 0 --p1-conf-threshold 0 \
    --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01 \
    --seed "${SEED}" --data "${DATA}" --resume --out "${RAW}" \
    >"${LOG}" 2>&1

rows=$(wc -l < "${RAW}")
if [[ ${rows} -ne 560 ]]; then
  echo "matched chain incomplete: ${rows}/560" >&2
  exit 4
fi
if rg -n "Traceback|CUDA error|RuntimeError|AssertionError" "${LOG}"; then
  echo "matched chain log contains a fatal signature" >&2
  exit 5
fi

"${PY}" "${ROOT}/experiments/proxy_async_overlap/tree_sweep/summarize_matched_chain_tree_20260813.py"
echo "[$(date -Is)] END matched chain: 560/560"
