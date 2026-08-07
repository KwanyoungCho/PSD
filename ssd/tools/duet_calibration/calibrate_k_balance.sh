#!/usr/bin/env bash
# Empirically balance DUET K1/K2 without a full grid sweep.
#
# 1. Profile the current pair once and linearly predict the local knee.
# 2. Measure predicted K1 and its two neighbours with K2 fixed.
# 3. At the selected K1, measure predicted K2 and its two neighbours.
# 4. Write k_balance.env only when every measured arm has enough aligned steps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MODE="${CALIB_MODE:-tree}"
case "${MODE}" in
  chain|tree) ;;
  *) echo "CALIB_MODE must be chain or tree (got ${MODE})" >&2; exit 2 ;;
esac
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/k_balance_${MODE}_${STAMP}}"
if [[ "${RESUME:-0}" != "1" ]] && [[ -e "${OUT}" ]] \
    && [[ -n "$(find "${OUT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to append to existing non-empty OUT=${OUT}" >&2
  exit 2
fi
mkdir -p "${OUT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1

MODEL_PATH="${MODEL_PATH:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
TARGET_AWQ="${TARGET_AWQ:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_PATH="${DRAFT_PATH:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
DRAFT_AWQ="${DRAFT_AWQ:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

BASE_K1="${CALIB_BASE_K1:-9}"
BASE_K2="${CALIB_BASE_K2:-4}"
EXIT_LAYER="${CALIB_EXIT_LAYER:-56}"
P1_FANOUT="${CALIB_P1_FANOUT:-2}"
P1_TEMPLATE="${CALIB_P1_FANOUT_LIST_TEMPLATE:-2,2,2,2,2,2,1,1,1,1}"
P2_BUDGET="${CALIB_P2_BUDGET:-10}"
ROOTS="${CALIB_P2_ROOTS:-10}"
NV="${CALIB_NV:-8}"
C_TENSOR="${CALIB_C_TENSOR:-3}"
PROXY_THRESHOLD="${CALIB_PROXY_THRESHOLD:-0.01}"
CONF_THRESHOLD="${CALIB_CONF_THRESHOLD:-0.03}"
TEMP="${CALIB_TEMP:-0.7}"
NUMSEQS="${CALIB_NUMSEQS:-2}"
OUTLEN="${CALIB_OUTLEN:-128}"
INPUT_LEN="${CALIB_INPUT_LEN:-512}"
SEED="${CALIB_SEED:-42}"
BASE_PORT="${CALIB_BASE_PORT:-16800}"
MIN_STEPS="${CALIB_MIN_STEPS:-30}"
EXTRA=("$@")
ANALYZER="${ROOT}/tools/duet_calibration/analyze_k_balance.py"
RUN_SPECS=()
run_index=0
ACTIVE_PGID=""
BLOCK_SIZE="${CALIB_BLOCK_SIZE:-256}"
START_BUCKET=$(((INPUT_LEN + BASE_K1 + 1 + P2_BUDGET + BLOCK_SIZE - 1) / BLOCK_SIZE))

if [[ "${MODE}" == "tree" ]]; then
  TREE_EXEC="${CALIB_TREE_EXEC:-1}"
  TREE_ARENA="${CALIB_TREE_ARENA:-1}"
  TREE_PROXY_GRAPH="${CALIB_TREE_PROXY_GRAPH:-1}"
  # Capturing every page bucket into one executor is unsafe: later captures
  # can invalidate an earlier FlashInfer graph state.  Pre-capture only the
  # estimated first active bucket; callers may override it explicitly.
  TREE_WARMUP="${CALIB_TREE_WARMUP:-all}"
  EXIT_REPLICA="${CALIB_EXIT_REPLICA:-1}"
  ASYNC_PROXY_SEND="${CALIB_ASYNC_PROXY_SEND:-1}"
  POLICY_ARGS=(
    --duet_tree_policy "${CALIB_TREE_POLICY:-eagle}"
    --duet_tree_root_count "${ROOTS}" --duet_tree_nv "${NV}"
    --duet_tree_c_tensor "${C_TENSOR}"
    --duet_tree_fanout_policy backbone
    --duet_tree_proxy_threshold "${PROXY_THRESHOLD}"
    --duet_tree_conf_threshold "${CONF_THRESHOLD}")
else
  # These defaults reproduce the real chain path.  They can be overridden
  # explicitly when calibrating an experimental target/proxy transport path.
  TREE_EXEC="${CALIB_TREE_EXEC:-0}"
  TREE_ARENA="${CALIB_TREE_ARENA:-0}"
  TREE_PROXY_GRAPH="${CALIB_TREE_PROXY_GRAPH:-0}"
  TREE_WARMUP="${CALIB_TREE_WARMUP:-0}"
  EXIT_REPLICA="${CALIB_EXIT_REPLICA:-0}"
  ASYNC_PROXY_SEND="${CALIB_ASYNC_PROXY_SEND:-0}"
  POLICY_ARGS=(--duet_tree_policy off)
fi

cleanup_active_group () {
  if [[ -n "${ACTIVE_PGID}" ]]; then
    kill -TERM -- "-${ACTIVE_PGID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${ACTIVE_PGID}" 2>/dev/null || true
    ACTIVE_PGID=""
  fi
}
trap cleanup_active_group EXIT
trap 'exit 130' INT TERM

if (( BASE_K2 > BASE_K1 )); then
  echo "require CALIB_BASE_K1 >= CALIB_BASE_K2" >&2
  exit 2
fi
if [[ "${MODE}" == "tree" ]] && (( ROOTS > P2_BUDGET )); then
  echo "EAGLE requires CALIB_P2_ROOTS <= CALIB_P2_BUDGET" >&2
  exit 2
fi

p1_list_for () {
  local k1="$1"
  local values=()
  IFS=',' read -r -a values <<<"${P1_TEMPLATE}"
  if (( ${#values[@]} == 0 )); then
    echo "empty CALIB_P1_FANOUT_LIST_TEMPLATE" >&2
    return 2
  fi
  local last="${values[$((${#values[@]} - 1))]}" out="" i value
  for ((i=0; i<=k1; i++)); do
    value="${last}"
    if (( i < ${#values[@]} )); then value="${values[$i]}"; fi
    if [[ -n "${out}" ]]; then out+=","; fi
    out+="${value}"
  done
  printf '%s' "${out}"
}

run_one () {
  local tag="$1" k1="$2" k2="$3"
  if (( k1 < 1 || k2 < 1 || k2 > k1 )); then
    echo "skip invalid K1=${k1} K2=${k2}" >&2
    return 0
  fi
  local dir="${OUT}/${tag}_k1_${k1}_k2_${k2}"
  local log="${dir}/run.log"
  local p1_list port leader rc
  p1_list="$(p1_list_for "${k1}")"
  port=$((BASE_PORT + run_index))
  run_index=$((run_index + 1))
  if [[ "${RESUME:-0}" == "1" ]] \
      && grep -q '^EXIT:0$' "${log}" 2>/dev/null \
      && [[ -n "$(find "${dir}" -name 'duet_profile_draft_*.json' -print -quit 2>/dev/null)" ]] \
      && [[ -n "$(find "${dir}" -name 'duet_profile_target_rank0_*.json' -print -quit 2>/dev/null)" ]]; then
    echo "[$(date -Is)] reuse K1=${k1} K2=${k2} dir=${dir}"
    RUN_SPECS+=("--run" "${k1},${k2},${dir}")
    LAST_DIR="${dir}"
    return 0
  fi
  mkdir -p "${dir}"
  echo "[$(date -Is)] profile mode=${MODE} K1=${k1} K2=${k2} dir=${dir}"

  set +e
  setsid env \
    SSD_DIST_PORT="${port}" \
    SSD_PROFILE=0 SSD_PROFILE_DUET=1 SSD_PROFILE_DUET_DETAIL=0 \
    SSD_PROFILE_DUET_MAX_EVENTS=60000 SSD_PROFILE_DIR="${dir}" \
    SSD_TREE_EXEC="${TREE_EXEC}" SSD_TREE_ARENA="${TREE_ARENA}" \
    SSD_TREE_PROXY_GRAPH="${TREE_PROXY_GRAPH}" \
    SSD_TREE_EXEC_WARMUP="${TREE_WARMUP}" \
    SSD_DUET_EXIT_REPLICA="${EXIT_REPLICA}" \
    SSD_ASYNC_PROXY_SEND="${ASYNC_PROXY_SEND}" SSD_PROXY_STREAM=0 \
    timeout --kill-after=30s "${CALIB_TIMEOUT:-45m}" \
    "${PY}" -O bench/bench.py --llama --size 8 \
      --model_path "${MODEL_PATH}" --quant_awq \
      --quant_awq_artifact "${TARGET_AWQ}" --quant_group_size 128 \
      --b 1 --temp "${TEMP}" --input_len "${INPUT_LEN}" --all \
      --max_model_len 2048 --draft_path "${DRAFT_PATH}" \
      --quant_awq_draft --quant_awq_draft_artifact "${DRAFT_AWQ}" \
      --gpus 5 --async --spec --duet "${EXTRA[@]}" \
      --duet_exit_layer "${EXIT_LAYER}" \
      --f 3 --duet_k1 "${k1}" --duet_k2 "${k2}" \
      --duet_p1_fanout "${P1_FANOUT}" \
      --duet_p1_fanout_list "${p1_list}" \
      --duet_p2_budget "${P2_BUDGET}" "${POLICY_ARGS[@]}" \
      --seed "${SEED}" --numseqs "${NUMSEQS}" --output_len "${OUTLEN}" \
      >"${log}" 2>&1 &
  leader=$!
  ACTIVE_PGID="${leader}"
  wait "${leader}"
  rc=$?
  # torch.multiprocessing children can survive a rank-level CUDA abort.  The
  # exact setsid process group belongs to this arm, so always reap that group.
  kill -TERM -- "-${leader}" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-${leader}" 2>/dev/null || true
  ACTIVE_PGID=""
  set -e
  echo "EXIT:${rc}" >>"${log}"
  if (( rc != 0 )); then
    echo "K1=${k1} K2=${k2} failed rc=${rc}; see ${log}" >&2
    exit "${rc}"
  fi
  test -n "$(find "${dir}" -name 'duet_profile_draft_*.json' -print -quit)"
  test -n "$(find "${dir}" -name 'duet_profile_target_rank0_*.json' -print -quit)"
  RUN_SPECS+=("--run" "${k1},${k2},${dir}")
  LAST_DIR="${dir}"
}

unique_local_candidates () {
  local center="$1" minimum="$2" maximum="$3" explicit="$4" required="$5"
  local raw=() x seen=" "
  if [[ -n "${explicit}" ]]; then
    read -r -a raw <<<"${explicit}"
  else
    raw=($((center - 1)) "${center}" $((center + 1)))
  fi
  raw+=("${required}")
  for x in "${raw[@]}"; do
    if (( x < minimum || x > maximum )); then continue; fi
    if [[ "${seen}" != *" ${x} "* ]]; then
      printf '%s\n' "${x}"
      seen+="${x} "
    fi
  done
}

# Baseline: its signed gaps predict a three-point local K1 search.
run_one baseline "${BASE_K1}" "${BASE_K2}"
BASE_DIR="${LAST_DIR}"
"${PY}" "${ANALYZER}" --run "${BASE_K1},${BASE_K2},${BASE_DIR}" \
  --min-steps "${MIN_STEPS}" --json-out "${OUT}/baseline.json" \
  >"${OUT}/baseline.txt"
PRED_K1="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][0]["predicted_k1"])' "${OUT}/baseline.json")"

mapfile -t K1_VALUES < <(unique_local_candidates \
  "${PRED_K1}" "${BASE_K2}" 32 "${CALIB_K1_VALUES:-}" "${BASE_K1}")
K1_SPECS=()
for k1 in "${K1_VALUES[@]}"; do
  if (( k1 == BASE_K1 )); then
    K1_SPECS+=("--run" "${k1},${BASE_K2},${BASE_DIR}")
  else
    run_one stage1 "${k1}" "${BASE_K2}"
    K1_SPECS+=("--run" "${k1},${BASE_K2},${LAST_DIR}")
  fi
done
"${PY}" "${ANALYZER}" "${K1_SPECS[@]}" --min-steps "${MIN_STEPS}" \
  --json-out "${OUT}/stage1_k1.json" --strict | tee "${OUT}/stage1_k1.txt"
SELECTED_K1="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommendation"]["k1"])' "${OUT}/stage1_k1.json")"

# Get the local K2 prediction from the measured selected-K1/base-K2 arm.
SELECTED_BASE_DIR=""
for ((i=1; i<${#K1_SPECS[@]}; i+=2)); do
  spec="${K1_SPECS[$i]}"
  IFS=',' read -r sk1 sk2 spath <<<"${spec}"
  if (( sk1 == SELECTED_K1 )); then SELECTED_BASE_DIR="${spath}"; break; fi
done
test -n "${SELECTED_BASE_DIR}"
"${PY}" "${ANALYZER}" --run \
  "${SELECTED_K1},${BASE_K2},${SELECTED_BASE_DIR}" \
  --min-steps "${MIN_STEPS}" --json-out "${OUT}/k2_prediction.json" \
  >"${OUT}/k2_prediction.txt"
PRED_K2="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][0]["predicted_k2"])' "${OUT}/k2_prediction.json")"

mapfile -t K2_VALUES < <(unique_local_candidates \
  "${PRED_K2}" 1 "${SELECTED_K1}" "${CALIB_K2_VALUES:-}" "${BASE_K2}")
K2_SPECS=()
for k2 in "${K2_VALUES[@]}"; do
  if (( k2 == BASE_K2 )); then
    K2_SPECS+=("--run" "${SELECTED_K1},${k2},${SELECTED_BASE_DIR}")
  else
    run_one stage2 "${SELECTED_K1}" "${k2}"
    K2_SPECS+=("--run" "${SELECTED_K1},${k2},${LAST_DIR}")
  fi
done
"${PY}" "${ANALYZER}" "${K2_SPECS[@]}" \
  --preferred-k1 "${SELECTED_K1}" --min-steps "${MIN_STEPS}" \
  --json-out "${OUT}/final.json" --strict | tee "${OUT}/final.txt"
SELECTED_K2="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommendation"]["k2"])' "${OUT}/final.json")"
printf '# Generated by calibrate_k_balance.sh\nCALIB_MODE=%s\nDUET_K1=%s\nDUET_K2=%s\n' \
  "${MODE}" "${SELECTED_K1}" "${SELECTED_K2}" >"${OUT}/k_balance.env"

echo "[$(date -Is)] K balance complete: mode=${MODE} K1=${SELECTED_K1} K2=${SELECTED_K2}"
echo "  report: ${OUT}/final.txt"
echo "  config: ${OUT}/k_balance.env"
