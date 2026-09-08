#!/usr/bin/env bash
# Full Spec-Bench local sweep around the current DUET P1+P2 tree configuration.
# Every arm uses all 480 questions / 560 turns and the same sampler seed.
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
DATA=${BASE}/data/specbench_full.jsonl
OUT=${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_full_local_sweep_seed42_20260812
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU_SET=${GPU_SET:-6,7,5}
SEED=42
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
  --template raw --seed "${SEED}" --warmup 2
  --p1-tree on --p2-tree on --p1-allocation-policy backbone
  --roots-per-position 3 --root-count 10
  --n2 8 --p2-verify-nodes 8
  --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01
  --data "${DATA}"
)

check_gpus() {
  local gpu used
  IFS=',' read -ra selected <<< "${GPU_SET}"
  for gpu in "${selected[@]}"; do
    used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "${used}" -gt 2000 ]]; then
      echo "GPU ${gpu} is not free (${used} MiB used)" >&2
      return 1
    fi
  done
}

selected() {
  local arm=$1
  [[ ",${ARMS:-reference_repeat,n1_12,c3,threshold_mild}," == *",${arm},"* ]]
}

run_arm() {
  local arm=$1 c=$2 n1=$3 m1=$4 start_thr=$5 conf_thr=$6 port=$7
  local raw=${OUT}/${arm}_s${SEED}_o${OUTLEN}.jsonl
  local log=${OUT}/${arm}_s${SEED}_o${OUTLEN}.log
  local rows=0
  [[ -e "${raw}" ]] && rows=$(wc -l < "${raw}")
  if [[ ${rows} -eq 560 ]]; then
    echo "[$(date -Is)] SKIP complete ${arm}"
    "${PY}" "${ROOT}/experiments/proxy_async_overlap/tree_sweep/summarize_full_local_sweep.py"
    return 0
  fi
  if [[ ${rows} -gt 560 ]]; then
    echo "invalid row count for ${raw}: ${rows}" >&2
    return 3
  fi
  check_gpus
  echo "[$(date -Is)] START ${arm}: C=${c} N1=${n1} M1=${m1} threshold=${start_thr}/${conf_thr} resume_rows=${rows}"
  SSD_DIST_PORT="${port}" timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
    "${PY}" -O "${RUNNER}" "${COMMON[@]}" \
      --c-tensor "${c}" --n1 "${n1}" --p1-verify-nodes "${m1}" \
      --p1-start-threshold "${start_thr}" --p1-conf-threshold "${conf_thr}" \
      --resume --out "${raw}" > "${log}" 2>&1
  rows=$(wc -l < "${raw}")
  if [[ ${rows} -ne 560 ]]; then
    echo "${arm} incomplete: ${rows}/560" >&2
    return 4
  fi
  echo "[$(date -Is)] END ${arm}: 560/560"
  "${PY}" "${ROOT}/experiments/proxy_async_overlap/tree_sweep/summarize_full_local_sweep.py"
}

# Reference repeat quantifies independent-run variance at the exact current config.
selected reference_repeat && run_arm reference_repeat 2 14 12 0 0 18520
# N1=M1 removes the on-hit 14->12 rerank while keeping the target verify cap.
selected n1_12            && run_arm n1_12            2 12 12 0 0 18521
# Other one-factor-at-a-time neighbors. M1 stays 12 in every arm.
selected c3               && run_arm c3               3 14 12 0 0 18522
# Restores the earlier mild P1 pruning thresholds; all other parameters stay fixed.
selected threshold_mild   && run_arm threshold_mild   2 14 12 0.001 0.01 18523

echo "[$(date -Is)] full local sweep complete: ${OUT}"
