#!/usr/bin/env bash
# Profiler-off paired screening run on the fixed hold-out subset.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
TAG=${1:?usage: run_screen_arm.sh TAG}
ARM=${HERE}/screen/${TAG}
DATA=${DATA:-${HERE}/screening_subset.jsonl}
GPU_SET=${GPU_SET:-5,6,7}
PORT=${PORT:-18940}

K1=${K1:-8}; K2=${K2:-4}; EXIT_LAYER=${EXIT_LAYER:-56}
N1=${N1:-14}; M1=${M1:-12}; N2=${N2:-8}; M2=${M2:-8}
RPP=${RPP:-3}; ROOT_COUNT=${ROOT_COUNT:-10}; C_TENSOR=${C_TENSOR:-2}
P1_START=${P1_START:-0}; P1_CONF=${P1_CONF:-0}
P2_PROXY=${P2_PROXY:-0.01}; P2_CONF=${P2_CONF:-0.01}
OUTLEN=${OUTLEN:-512}; SEED=${SEED:-42}

if [[ -e ${ARM}/run.log || -e ${ARM}/raw.jsonl ]]; then
  echo "refusing to append to existing arm: ${ARM}" >&2
  exit 2
fi
mkdir -p "${ARM}"

IFS=',' read -ra GPUS <<<"${GPU_SET}"
for gpu in "${GPUS[@]}"; do
  used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( used > 2000 )); then
    echo "GPU ${gpu} is not free (${used} MiB)" >&2
    exit 3
  fi
done

export DUET_ROOT=${ROOT} CUDA_VISIBLE_DEVICES=${GPU_SET}
export HF_HOME=/home/eslab/models SSD_HF_CACHE=/home/eslab/models/hub
export SSD_DATASET_DIR=${BASE}/data TORCH_CUDA_ARCH_LIST=12.0 SSD_CUDA_ARCH=12.0
export SSD_ATTN_BACKEND=auto SSD_CHAIN_PROXY_GRAPH=1
export SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0
export SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 SSD_TREE_PROXY_GRAPH=1
export SSD_TREE_EXEC_WARMUP=all SSD_TREE_VERIFY_WORKSPACE_MB=224
export SSD_TREE_EXEC_WORKSPACE_MB=128 SSD_P1_TREE_EXEC_WORKSPACE_MB=128
export SSD_TREE_TOPOLOGY_GPU=1 SSD_P1_RERANK_PRECOMPUTE=1
export SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0
export SSD_DUET_E0_TRACE=0
unset SSD_PROFILE_DIR SSD_TREE_TOPO_TRACE SSD_TREE_CALIB_TRACE
unset SSD_TREE_NODE_AUDIT SSD_TREE_STAGE1 SSD_TREE_STAGE2

echo "[$(date -Is)] screen ${TAG}: K=${K1}/${K2} exit=${EXIT_LAYER} " \
     "N/M=${N1}/${M1},${N2}/${M2} R=${ROOT_COUNT} RPP=${RPP}" | tee "${ARM}/run.log"

set +e
SSD_DIST_PORT=${PORT} timeout --signal=TERM --kill-after=60s 180m \
  "${PY}" -O "${RUNNER}" \
    --target facebook/layerskip-llama2-70B \
    --draft TinyLlama/TinyLlama-1.1B-Chat-v1.0 --gpus 3 \
    --k1 "${K1}" --k2 "${K2}" --exit-layer "${EXIT_LAYER}" \
    --p1-fanout 3 --p2-budget 15 --proxy-top-k 28 \
    --temp 0.7 --top_p 1.0 --max_new_tokens "${OUTLEN}" \
    --max_model_len 2048 --template raw --warmup 2 \
    --p1-tree on --p2-tree on --p1-allocation-policy backbone \
    --roots-per-position "${RPP}" --root-count "${ROOT_COUNT}" \
    --c-tensor "${C_TENSOR}" --n1 "${N1}" --p1-verify-nodes "${M1}" \
    --n2 "${N2}" --p2-verify-nodes "${M2}" \
    --p1-start-threshold "${P1_START}" --p1-conf-threshold "${P1_CONF}" \
    --p2-proxy-threshold "${P2_PROXY}" --p2-conf-threshold "${P2_CONF}" \
    --seed "${SEED}" --data "${DATA}" --out "${ARM}/raw.jsonl" \
    >>"${ARM}/run.log" 2>&1
rc=$?
set -e
echo "EXIT:${rc}" >>"${ARM}/run.log"
if (( rc != 0 )); then exit "${rc}"; fi
if rg -n "FATAL: draft runner|Traceback \(most recent call last\)|CUDA error|AssertionError" \
    "${ARM}/run.log" >"${ARM}/fatal_scan.txt"; then
  echo "engine failure found despite runner exit code 0; see fatal_scan.txt" >&2
  exit 5
fi
expected=$(wc -l <"${DATA}")
actual=$(wc -l <"${ARM}/raw.jsonl")
if (( actual != expected )); then
  echo "incomplete raw output: expected ${expected}, got ${actual}" >&2
  exit 6
fi
"${PY}" "${BASE}/analysis/question_level_metrics.py" \
  "${ARM}/raw.jsonl" --json >"${ARM}/metrics.json"
echo "[$(date -Is)] screen ${TAG} complete"
