#!/usr/bin/env bash
# Fair-root P1 chain/tree audit on the seven-category Spec-Bench tiny set.
#
# U1 holds both cache-root coverage and draft forward cells fixed:
#   contexts = K1 + 1 = 8
#   chain roots = 8 * fanout1
#   tree roots  = 8 * rpp1
#   forward cells = 8 roots * K1(7 rounds) = 56
# C=1 is the chain-degenerate contract.  C=3 may redistribute those same
# 56 cells into siblings / stronger roots.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${BASE:-/home/eslab/chokwans99/baseline}"
PY="${PY:-${BASE}/.venv-ssd/bin/python}"
RUNNER="${BASE}/runners/run_duet.py"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_tree_audit_20260810}"
DATA="${DATA:-${BASE}/data/specbench_tiny.jsonl}"
TARGET="${TARGET:-facebook/layerskip-llama2-70B}"
DRAFT="${DRAFT:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
GPU_SET="${GPU_SET:-6,7,5}"
SEED="${SEED:-42}"
OUTLEN="${OUTLEN:-128}"
K1="${K1:-7}"
K2="${K2:-4}"
P2_BUDGET="${P2_BUDGET:-10}"
PROXY_TOP_K="${PROXY_TOP_K:-90}"
N1="${N1:-14}"
P1_VERIFY="${P1_VERIFY:-${N1}}"

mkdir -p "${OUT}"
cd "${BASE}"

export DUET_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export HF_HOME="${HF_HOME:-/home/eslab/models}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/eslab/models/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-${BASE}/data}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-12.0}"
export SSD_ATTN_BACKEND="${SSD_ATTN_BACKEND:-auto}"
export SSD_CHAIN_PROXY_GRAPH=1
export SSD_DUET_EXIT_REPLICA=1
export SSD_ASYNC_PROXY_SEND=1
export SSD_PROXY_STREAM=0
export SSD_TREE_ARENA=1
export SSD_TREE_PROXY_GRAPH=1
export SSD_TREE_VERIFY_WORKSPACE_MB="${SSD_TREE_VERIFY_WORKSPACE_MB:-224}"
export SSD_TREE_EXEC_WORKSPACE_MB="${SSD_TREE_EXEC_WORKSPACE_MB:-128}"
export SSD_P1_TREE_EXEC_WORKSPACE_MB="${SSD_P1_TREE_EXEC_WORKSPACE_MB:-128}"
export SSD_PROFILE=0
export SSD_PROFILE_DUET=0
export SSD_PROFILE_DUET_DETAIL=0

# Correctness traces and performance measurements must never be mixed.  A
# caller may request a separate diagnostic run with TRACE_PREFIX; its TPS is
# deliberately excluded from the performance summary.
if [[ -n "${TRACE_PREFIX:-}" ]]; then
  mkdir -p "$(dirname "${TRACE_PREFIX}")" "${OUT}/e0"
  export SSD_TREE_TOPO_TRACE="${TRACE_PREFIX}"
  export SSD_TREE_NODE_AUDIT="${TRACE_PREFIX}.nodes"
  export SSD_TREE_CALIB_TRACE="${TRACE_PREFIX}.calib"
  export SSD_DUET_E0_TRACE=1 SSD_DUET_E0_SUBSAMPLE=1
  export SSD_DUET_E0_DIR="${OUT}/e0"
else
  unset SSD_TREE_TOPO_TRACE SSD_TREE_NODE_AUDIT SSD_TREE_CALIB_TRACE
  unset SSD_DUET_E0_TRACE SSD_DUET_E0_DIR
fi

COMMON=(
  --target "${TARGET}" --draft "${DRAFT}" --gpus 3
  --k1 "${K1}" --k2 "${K2}" --exit-layer 56
  --p1-fanout 1 --p2-budget "${P2_BUDGET}"
  --proxy-top-k "${PROXY_TOP_K}"
  --temp 0.7 --top_p 1.0
  --max_new_tokens "${OUTLEN}" --max_model_len 4096
  --extend-draft-rope --template raw --seed "${SEED}" --warmup 1
  --p2-tree off --n2 8 --p2-verify-nodes 8
  --roots-per-position 1 --root-count 10
  --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01
  --p1-start-threshold 0 --p1-conf-threshold 0
  --data "${DATA}"
)

selected () {
  local arm="$1"
  [[ ",${ARMS:-u1_chain,u1_degenerate,u1_tree_c3}," == *",${arm},"* ]]
}

run_arm () {
  local arm="$1" p1_tree="$2" c="$3" port="$4"
  local allocation="${5:-dynamic}"
  local json="${OUT}/${arm}_s${SEED}_o${OUTLEN}.jsonl"
  local log="${OUT}/${arm}_s${SEED}_o${OUTLEN}.log"
  if [[ -s "${json}" && "${RESUME:-0}" == "1" ]]; then
    echo "SKIP ${arm}: ${json} exists"
    return 0
  fi
  if [[ "${p1_tree}" == "on" ]]; then
    export SSD_TREE_EXEC=1 SSD_TREE_EXEC_WARMUP=all
  else
    export SSD_TREE_EXEC=0 SSD_TREE_EXEC_WARMUP=0
  fi
  echo "[$(date -Is)] START ${arm} seed=${SEED} outlen=${OUTLEN}"
  SSD_DIST_PORT="${port}" timeout --kill-after=30s 45m \
    "${PY}" -O "${RUNNER}" "${COMMON[@]}" \
      --p1-tree "${p1_tree}" --c-tensor "${c}" \
      --p1-allocation-policy "${allocation}" \
      --n1 "${N1}" --p1-verify-nodes "${P1_VERIFY}" --out "${json}" \
      >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  echo "[$(date -Is)] END ${arm} rc=${rc} rows=$(wc -l < "${json}" 2>/dev/null || echo 0)"
  if [[ ${rc} -ne 0 ]]; then
    tail -n 30 "${log}"
    return "${rc}"
  fi
}

if selected u1_chain; then
  run_arm u1_chain off 1 18110 dynamic || exit $?
fi
if selected u1_degenerate; then
  run_arm u1_degenerate on 1 18111 dynamic || exit $?
fi
if selected u1_tree_c3; then
  run_arm u1_tree_c3 on 3 18112 dynamic || exit $?
fi
if selected u1_backbone; then
  run_arm u1_backbone on 3 18113 backbone || exit $?
fi
if selected u1_hybrid; then
  run_arm u1_hybrid on 3 18114 hybrid || exit $?
fi

"${PY}" "${ROOT}/experiments/proxy_async_overlap/tree_sweep/summarize_p1_tree_audit.py" \
  "${OUT}"/*_s"${SEED}"_o"${OUTLEN}".jsonl \
  --csv "${OUT}/summary_s${SEED}_o${OUTLEN}.csv"
