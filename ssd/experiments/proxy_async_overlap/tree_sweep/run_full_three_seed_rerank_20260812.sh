#!/usr/bin/env bash
# Full Spec-Bench validation of the fused P1 rerank path at three seeds.
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
DATA=${BASE}/data/specbench_full.jsonl
OUT=${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_full_rerank_3seed_20260812
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU_SET=${GPU_SET:-5,6,7}
SEEDS=${SEEDS:-"1 42 123"}
OUTLEN=1024
TIMEOUT_MIN=${TIMEOUT_MIN:-180}

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
export SSD_TREE_EXEC=1
export SSD_TREE_ARENA=1
export SSD_TREE_PROXY_GRAPH=1
export SSD_TREE_EXEC_WARMUP=all
export SSD_TREE_VERIFY_WORKSPACE_MB=224
export SSD_TREE_EXEC_WORKSPACE_MB=128
export SSD_P1_TREE_EXEC_WORKSPACE_MB=128
export SSD_TREE_TOPOLOGY_GPU=1
export SSD_P1_RERANK_PRECOMPUTE=1
export SSD_PROFILE=0
export SSD_PROFILE_DUET=0
export SSD_PROFILE_DUET_DETAIL=0

if [[ $(wc -l < "${DATA}") -ne 560 ]]; then
  echo "expected 560-turn full Spec-Bench input: ${DATA}" >&2
  exit 2
fi

COMMON=(
  --target "${TARGET}" --draft "${DRAFT}" --gpus 3
  --k1 8 --k2 4 --exit-layer 56
  --p1-fanout 3 --p2-budget 15 --proxy-top-k 28
  --temp 0.7 --top_p 1.0 --max_new_tokens "${OUTLEN}"
  --max_model_len 4096 --extend-draft-rope
  --template raw --warmup 2
  --p1-tree on --p2-tree on --p1-allocation-policy backbone
  --roots-per-position 3 --root-count 10 --c-tensor 2
  --n1 14 --p1-verify-nodes 12 --n2 8 --p2-verify-nodes 8
  --p1-start-threshold 0 --p1-conf-threshold 0
  --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01
  --data "${DATA}"
)

check_gpus() {
  local gpu used
  IFS=',' read -ra selected <<< "${GPU_SET}"
  for gpu in "${selected[@]}"; do
    used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used \
      --format=csv,noheader,nounits)
    if [[ "${used}" -gt 2000 ]]; then
      echo "GPU ${gpu} is not free (${used} MiB used)" >&2
      return 1
    fi
  done
}

run_seed() {
  local seed=$1 port=$2
  local stem=duet_tree_rerank_s${seed}_o${OUTLEN}
  local raw=${OUT}/${stem}.jsonl
  local log=${OUT}/${stem}.log
  local rows=0
  [[ -e "${raw}" ]] && rows=$(wc -l < "${raw}")
  if [[ ${rows} -eq 560 ]]; then
    echo "[$(date -Is)] SKIP complete seed=${seed}"
    return 0
  fi
  if [[ ${rows} -gt 560 ]]; then
    echo "invalid row count ${rows}: ${raw}" >&2
    return 3
  fi
  check_gpus
  echo "[$(date -Is)] START seed=${seed} resume_rows=${rows}"
  SSD_DIST_PORT="${port}" \
    timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
    "${PY}" -O "${RUNNER}" "${COMMON[@]}" --seed "${seed}" \
      --resume --out "${raw}" >"${log}" 2>&1
  rows=$(wc -l < "${raw}")
  if [[ ${rows} -ne 560 ]]; then
    echo "seed ${seed} incomplete: ${rows}/560" >&2
    return 4
  fi
  if rg -n "Traceback|CUDA error|RuntimeError|AssertionError" "${log}"; then
    echo "seed ${seed} log contains a fatal signature" >&2
    return 5
  fi
  echo "[$(date -Is)] END seed=${seed}: 560/560"
}

port=18710
for seed in ${SEEDS}; do
  run_seed "${seed}" "${port}"
  port=$((port + 1))
done

echo "[$(date -Is)] all three full-data seeds complete: ${OUT}"
