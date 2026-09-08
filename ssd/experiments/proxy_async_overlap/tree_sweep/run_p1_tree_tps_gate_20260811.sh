#!/usr/bin/env bash
# P1-tree AL/TPS gate against the paper P2-tree-only configuration.
#
# The arms use one dataset, seed, output limit, proxy candidate width, cache
# root budgets, and P2 tree.  Therefore P1 conditional accepted length and
# target verification time isolate the P1 topology/allocation change.  The
# common proxy top-k is explicit because enabling P1 tree otherwise raises it
# automatically (22 -> 28 in this shape), which would change the cache-root
# policy and confound the topology comparison.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${BASE:-/home/eslab/chokwans99/baseline}"
PY="${PY:-${BASE}/.venv-ssd/bin/python}"
RUNNER="${BASE}/runners/run_duet.py"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_tree_tps_gate_20260811}"
DATA="${DATA:-${BASE}/data/specbench_smoke.jsonl}"
TARGET="${TARGET:-facebook/layerskip-llama2-70B}"
DRAFT="${DRAFT:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
GPU_SET="${GPU_SET:-6,7,5}"
SEED="${SEED:-42}"
OUTLEN="${OUTLEN:-1024}"
K1="${K1:-8}"
K2="${K2:-4}"
P1_FANOUT="${P1_FANOUT:-3}"
P2_BUDGET="${P2_BUDGET:-15}"
PROXY_TOP_K="${PROXY_TOP_K:-28}"
RPP="${RPP:-3}"
C_TENSOR="${C_TENSOR:-3}"
N1="${N1:-14}"
P1_VERIFY="${P1_VERIFY:-12}"
N2="${N2:-8}"
P2_VERIFY="${P2_VERIFY:-8}"
TIMEOUT_MIN="${TIMEOUT_MIN:-60}"

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
export SSD_TREE_EXEC=1
export SSD_TREE_ARENA=1
export SSD_TREE_PROXY_GRAPH=1
export SSD_TREE_EXEC_WARMUP=all
export SSD_TREE_VERIFY_WORKSPACE_MB="${SSD_TREE_VERIFY_WORKSPACE_MB:-224}"
export SSD_TREE_EXEC_WORKSPACE_MB="${SSD_TREE_EXEC_WORKSPACE_MB:-128}"
export SSD_P1_TREE_EXEC_WORKSPACE_MB="${SSD_P1_TREE_EXEC_WORKSPACE_MB:-128}"
export SSD_PROFILE="${PROFILE:-0}"
export SSD_PROFILE_DUET="${PROFILE_DUET:-0}"
export SSD_PROFILE_DUET_DETAIL="${PROFILE_DETAIL:-0}"

COMMON=(
  --target "${TARGET}" --draft "${DRAFT}" --gpus 3
  --k1 "${K1}" --k2 "${K2}" --exit-layer 56
  --p1-fanout "${P1_FANOUT}" --p2-budget "${P2_BUDGET}"
  --proxy-top-k "${PROXY_TOP_K}"
  --temp 0.7 --top_p 1.0
  --max_new_tokens "${OUTLEN}" --max_model_len 4096
  --extend-draft-rope --template raw --seed "${SEED}" --warmup 2
  --p2-tree on --n2 "${N2}" --p2-verify-nodes "${P2_VERIFY}"
  --roots-per-position "${RPP}" --root-count 10
  --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01
  --p1-start-threshold 0 --p1-conf-threshold 0
  --data "${DATA}"
)

selected () {
  local arm="$1"
  [[ ",${ARMS:-p2_only,p1_dynamic,p1_backbone}," == *",${arm},"* ]]
}

run_arm () {
  local arm="$1" p1_tree="$2" allocation="$3" port="$4"
  local json="${OUT}/${arm}_s${SEED}_o${OUTLEN}.jsonl"
  local log="${OUT}/${arm}_s${SEED}_o${OUTLEN}.log"
  if [[ -s "${json}" && "${RESUME:-0}" == "1" ]]; then
    echo "SKIP ${arm}: ${json} exists"
    return 0
  fi
  echo "[$(date -Is)] START ${arm} seed=${SEED} outlen=${OUTLEN}"
  SSD_DIST_PORT="${port}" SSD_PROFILE_DIR="${OUT}/${arm}_profile" \
    timeout --kill-after=30s "${TIMEOUT_MIN}m" \
    "${PY}" -O "${RUNNER}" "${COMMON[@]}" \
      --p1-tree "${p1_tree}" --c-tensor "${C_TENSOR}" \
      --p1-allocation-policy "${allocation}" \
      --n1 "${N1}" --p1-verify-nodes "${P1_VERIFY}" --out "${json}" \
      >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  echo "[$(date -Is)] END ${arm} rc=${rc} rows=$(wc -l < "${json}" 2>/dev/null || echo 0)"
  if [[ ${rc} -ne 0 ]]; then
    tail -n 50 "${log}"
    return "${rc}"
  fi
}

if selected p2_only; then
  run_arm p2_only off dynamic 18310 || exit $?
fi
if selected p1_dynamic; then
  run_arm p1_dynamic on dynamic 18311 || exit $?
fi
if selected p1_backbone; then
  run_arm p1_backbone on backbone 18312 || exit $?
fi

"${PY}" "${ROOT}/experiments/proxy_async_overlap/tree_sweep/summarize_p1_tree_audit.py" \
  "${OUT}"/*_s"${SEED}"_o"${OUTLEN}".jsonl \
  --csv "${OUT}/summary_s${SEED}_o${OUTLEN}.csv"
