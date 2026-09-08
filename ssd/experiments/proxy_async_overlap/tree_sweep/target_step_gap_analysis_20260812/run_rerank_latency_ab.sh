#!/usr/bin/env bash
# Small, repeated latency-only A/B for the P1 tree on-hit rerank.
#
# N1=14,M1=12 is the current paper tree configuration.  N1=12,M1=12 keeps
# the number of rows verified by the target fixed while selecting the equal-
# limit fast path in _rerank_tree_hit_view.  The runs are alternated to avoid
# assigning machine drift to only one arm.  This is a diagnostic, not a TPS,
# AL, or hit-rate result.
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
DATA=${BASE}/data/specbench_tiny.jsonl
OUT=${ROOT}/experiments/proxy_async_overlap/tree_sweep/target_step_gap_analysis_20260812/rerank_ab
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU_SET=${GPU_SET:-6,7,5}
REPEATS=${REPEATS:-3}
TIMEOUT_MIN=${TIMEOUT_MIN:-35}

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
export SSD_PROFILE=0
export SSD_PROFILE_DUET=1
export SSD_PROFILE_DUET_DETAIL=1
export SSD_PROFILE_DUET_MAX_EVENTS=50000

if [[ $(wc -l < "${DATA}") -ne 7 ]]; then
  echo "expected the canonical seven-turn diagnostic input: ${DATA}" >&2
  exit 2
fi

COMMON=(
  --target "${TARGET}" --draft "${DRAFT}" --gpus 3
  --k1 8 --k2 4 --exit-layer 56
  --p1-fanout 3 --p2-budget 15 --proxy-top-k 28
  --temp 0.7 --top_p 1.0 --max_new_tokens 256
  --max_model_len 4096 --extend-draft-rope
  --template raw --seed 42 --warmup 2
  --p1-tree on --p2-tree on --p1-allocation-policy backbone
  --roots-per-position 3 --root-count 10 --c-tensor 2
  --p1-verify-nodes 12 --n2 8 --p2-verify-nodes 8
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

run_one() {
  local arm=$1 rep=$2 n1=$3 port=$4
  local stem=${arm}_r${rep}_s42_o256
  local raw=${OUT}/${stem}.jsonl
  local log=${OUT}/${stem}.log
  local profile=${OUT}/${stem}_profile

  if [[ -e "${raw}" || -e "${profile}" ]]; then
    echo "refusing to overwrite existing diagnostic: ${stem}" >&2
    return 3
  fi
  check_gpus
  mkdir -p "${profile}"
  echo "[$(date -Is)] START ${arm} repeat=${rep} N1=${n1} M1=12"
  SSD_DIST_PORT="${port}" SSD_PROFILE_DIR="${profile}" \
    timeout --signal=TERM --kill-after=30s "${TIMEOUT_MIN}m" \
    "${PY}" -O "${RUNNER}" "${COMMON[@]}" --n1 "${n1}" \
      --out "${raw}" >"${log}" 2>&1
  local rows
  rows=$(wc -l < "${raw}")
  if [[ "${rows}" -ne 7 ]]; then
    echo "incomplete diagnostic ${stem}: ${rows}/7" >&2
    return 4
  fi
  echo "[$(date -Is)] END ${arm} repeat=${rep}: 7/7"
}

for ((rep=1; rep<=REPEATS; rep++)); do
  # Alternate within each repeat.  Ports are deterministic and disjoint.
  run_one n14_m12 "${rep}" 14 "$((18540 + (rep - 1) * 2))"
  run_one n12_m12 "${rep}" 12 "$((18541 + (rep - 1) * 2))"
done

echo "[$(date -Is)] rerank latency A/B complete: ${OUT}"
