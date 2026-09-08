#!/usr/bin/env bash
# Repeat the paper DUET-chain configuration with two additional sampler seeds.
# The raw run matches the reported 4,096-context source; a fixed prompt-only
# filter then creates the common 2,048-token-safe 456-question artifact.
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd
BASE=/home/eslab/chokwans99/baseline
PAPER=/home/eslab/chokwans99/DUET_PAPER_RESULTS
PY=${BASE}/.venv-ssd/bin/python
RUNNER=${BASE}/runners/run_duet.py
DATA=${BASE}/data/specbench_full.jsonl
FILTER=${PAPER}/scripts/metrics/filter_context_safe_subset.py
METRICS=${PAPER}/scripts/metrics/question_level_metrics.py
OUT=${ROOT}/experiments/proxy_async_overlap/tree_sweep/chain_paper_config_seeds1_123_20260815
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
GPU_SET=${GPU_SET:-5,6,7}
TIMEOUT_MIN=${TIMEOUT_MIN:-300}
SEEDS=(${SEEDS:-1 123})

mkdir -p "${OUT}"
cd "${BASE}"

export DUET_ROOT="${ROOT}"
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

wait_for_gpus() {
  local gpu used busy
  while true; do
    busy=0
    IFS=',' read -ra selected <<< "${GPU_SET}"
    for gpu in "${selected[@]}"; do
      used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used \
        --format=csv,noheader,nounits)
      if [[ "${used}" -gt 2000 ]]; then
        busy=1
      fi
    done
    if [[ ${busy} -eq 0 ]]; then
      echo "[$(date -Is)] GPUs ${GPU_SET} are free"
      return 0
    fi
    echo "[$(date -Is)] waiting for GPUs ${GPU_SET}"
    sleep 30
  done
}

for seed in "${SEEDS[@]}"; do
  stem=duet_chain_papercfg_s${seed}_o1024_ctx4096
  raw=${OUT}/${stem}.jsonl
  safe=${OUT}/${stem}_ctx2048_safe.jsonl
  log=${OUT}/${stem}.log
  rows=0
  [[ -f "${raw}" ]] && rows=$(wc -l < "${raw}")
  if [[ ${rows} -gt 560 ]]; then
    echo "invalid row count ${rows}: ${raw}" >&2
    exit 3
  fi
  if [[ ${rows} -lt 560 ]]; then
    wait_for_gpus
    export CUDA_VISIBLE_DEVICES="${GPU_SET}"
    port=$((18830 + seed % 1000))
    echo "[$(date -Is)] START chain seed=${seed} resume_rows=${rows}"
    SSD_DIST_PORT="${port}" \
      timeout --signal=TERM --kill-after=60s "${TIMEOUT_MIN}m" \
      "${PY}" -O "${RUNNER}" \
        --target "${TARGET}" --draft "${DRAFT}" --gpus 3 \
        --k1 8 --k2 4 --exit-layer 56 \
        --p1-fanout 3 --p2-budget 15 --proxy-top-k 28 \
        --temp 0.7 --top_p 1.0 --max_new_tokens 1024 \
        --max_model_len 4096 --extend-draft-rope --allow-nonpaper-context \
        --template raw --warmup 2 \
        --p1-tree off --p2-tree off --p1-allocation-policy backbone \
        --roots-per-position 3 --root-count 10 --c-tensor 2 \
        --n1 14 --p1-verify-nodes 12 --n2 8 --p2-verify-nodes 8 \
        --p1-start-threshold 0 --p1-conf-threshold 0 \
        --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01 \
        --seed "${seed}" --data "${DATA}" --resume --out "${raw}" \
        >"${log}" 2>&1
  fi

  rows=$(wc -l < "${raw}")
  if [[ ${rows} -ne 560 ]]; then
    echo "chain seed ${seed} incomplete: ${rows}/560" >&2
    exit 4
  fi
  if rg -n "Traceback|CUDA error|RuntimeError|AssertionError" "${log}"; then
    echo "chain seed ${seed} log contains a fatal signature" >&2
    exit 5
  fi
  "${PY}" "${FILTER}" "${raw}" "${safe}" \
    >"${OUT}/${stem}_ctx2048_filter.json"
  "${PY}" "${METRICS}" --json "${raw}" "${safe}" \
    >"${OUT}/${stem}_metrics.json"
  echo "[$(date -Is)] END chain seed=${seed}: raw=560, safe=$(wc -l < "${safe}")"
done

"${PY}" "${METRICS}" --json \
  "${OUT}"/*_ctx2048_safe.jsonl >"${OUT}/summary_ctx2048_safe.json"
echo "[$(date -Is)] COMPLETE all chain seeds"
