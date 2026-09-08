#!/usr/bin/env bash
# Latency-only controls that separate fixed tree machinery from extra rows.
# Existing full-gap triad supplies chain (P1=8/P2=4 chain rows) and P2-tree=8.
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
DATA=${BASE}/data/specbench_tiny.jsonl
OUT=${ROOT}/experiments/proxy_async_overlap/tree_sweep/target_step_gap_analysis_20260812/matched_width_controls
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU_SET=${GPU_SET:-6,7,5}
REPEATS=${REPEATS:-2}
TIMEOUT_MIN=${TIMEOUT_MIN:-35}

mkdir -p "${OUT}"
cd "${BASE}"

export DUET_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export HF_HOME=/home/eslab/models
export SSD_HF_CACHE=/home/eslab/models/hub
export SSD_DATASET_DIR=${BASE}/data
export TORCH_CUDA_ARCH_LIST=12.0 SSD_CUDA_ARCH=12.0 SSD_ATTN_BACKEND=auto
export SSD_CHAIN_PROXY_GRAPH=1 SSD_DUET_EXIT_REPLICA=1
export SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0
export SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 SSD_TREE_PROXY_GRAPH=1
export SSD_TREE_EXEC_WARMUP=all
export SSD_TREE_VERIFY_WORKSPACE_MB=224 SSD_TREE_EXEC_WORKSPACE_MB=128
export SSD_P1_TREE_EXEC_WORKSPACE_MB=128
export SSD_PROFILE=0 SSD_PROFILE_DUET=1 SSD_PROFILE_DUET_DETAIL=1
export SSD_PROFILE_DUET_MAX_EVENTS=70000

COMMON=(
  --target "${TARGET}" --draft "${DRAFT}" --gpus 3
  --k1 8 --k2 4 --exit-layer 56
  --p1-fanout 3 --p2-budget 15 --proxy-top-k 28
  --temp 0.7 --top_p 1.0 --max_new_tokens 256
  --max_model_len 4096 --extend-draft-rope
  --template raw --seed 42 --warmup 2
  --p1-allocation-policy backbone
  --roots-per-position 3 --root-count 10 --c-tensor 2
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
    [[ "${used}" -le 2000 ]] || {
      echo "GPU ${gpu} is not free (${used} MiB used)" >&2; return 1; }
  done
}

run_one() {
  local arm=$1 rep=$2 p1=$3 p2=$4 n1=$5 m1=$6 n2=$7 m2=$8 port=$9
  local stem=${arm}_r${rep}_s42_o256
  local raw=${OUT}/${stem}.jsonl log=${OUT}/${stem}.log
  local profile=${OUT}/${stem}_profile
  [[ ! -e "${raw}" && ! -e "${profile}" ]] || {
    echo "refusing to overwrite ${stem}" >&2; return 3; }
  check_gpus
  mkdir -p "${profile}"
  echo "[$(date -Is)] START ${arm} repeat=${rep}"
  SSD_DIST_PORT="${port}" SSD_PROFILE_DIR="${profile}" \
    timeout --signal=TERM --kill-after=30s "${TIMEOUT_MIN}m" \
    "${PY}" -O "${RUNNER}" "${COMMON[@]}" \
      --p1-tree "${p1}" --p2-tree "${p2}" \
      --n1 "${n1}" --p1-verify-nodes "${m1}" \
      --n2 "${n2}" --p2-verify-nodes "${m2}" --out "${raw}" \
      >"${log}" 2>&1
  [[ $(wc -l < "${raw}") -eq 7 ]] || return 4
  echo "[$(date -Is)] END ${arm} repeat=${rep}: 7/7"
}

for ((rep=1; rep<=REPEATS; rep++)); do
  port=$((18580 + (rep - 1) * 3))
  # P1 chain verifies 8 proposals (9 target rows); match that exactly.
  run_one p1_tree8  "${rep}" on off 8 8 8 8 "${port}"
  # Same fixed tree machinery, four additional verified nodes.
  run_one p1_tree12 "${rep}" on off 12 12 8 8 "$((port + 1))"
  # P2 chain verifies 4 proposals (5 target rows); match that exactly.
  run_one p2_tree4  "${rep}" off on 14 12 4 4 "$((port + 2))"
done

echo "[$(date -Is)] matched-width controls complete: ${OUT}"
