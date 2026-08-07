#!/usr/bin/env bash
# Profiler-OFF paired TPS/quality verification for chain or tree K candidates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
MODE="${CALIB_MODE:-tree}"
case "${MODE}" in chain|tree) ;; *) echo "CALIB_MODE must be chain or tree" >&2; exit 2;; esac
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/tps_${MODE}_${STAMP}}"
if [[ "${RESUME:-0}" != 1 && -e "${OUT}" ]] \
    && [[ -n "$(find "${OUT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing non-empty OUT=${OUT}; set RESUME=1 to reuse completed arms" >&2
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
K_CANDIDATES="${CALIB_K_CANDIDATES:-9:4}"
SEEDS="${CALIB_SEEDS:-42 123 2024}"
NUMSEQS="${CALIB_NUMSEQS:-10}"
OUTLEN="${CALIB_OUTLEN:-384}"
INPUT_LEN="${CALIB_INPUT_LEN:-512}"
TEMP="${CALIB_TEMP:-0.7}"
EXIT_LAYER="${CALIB_EXIT_LAYER:-56}"
P1_FANOUT="${CALIB_P1_FANOUT:-2}"
P1_TEMPLATE="${CALIB_P1_FANOUT_LIST_TEMPLATE:-2,2,2,2,2,2,1,1,1,1}"
P2_BUDGET="${CALIB_P2_BUDGET:-10}"
ROOTS="${CALIB_P2_ROOTS:-10}"
NV="${CALIB_NV:-8}"
C_TENSOR="${CALIB_C_TENSOR:-3}"
PROXY_THRESHOLD="${CALIB_PROXY_THRESHOLD:-0.01}"
CONF_THRESHOLD="${CALIB_CONF_THRESHOLD:-0.03}"
BASE_PORT="${CALIB_BASE_PORT:-16900}"
EXTRA=("$@")
BLOCK_SIZE="${CALIB_BLOCK_SIZE:-256}"
MAX_K1=1
for _candidate in ${K_CANDIDATES}; do
  IFS=':' read -r _k1 _k2 <<<"${_candidate}"
  if [[ "${_k1}" =~ ^[0-9]+$ ]] && (( _k1 > MAX_K1 )); then
    MAX_K1="${_k1}"
  fi
done
START_BUCKET=$(((INPUT_LEN + MAX_K1 + 1 + P2_BUDGET + BLOCK_SIZE - 1) / BLOCK_SIZE))

if [[ "${MODE}" == tree ]]; then
  TREE_EXEC="${CALIB_TREE_EXEC:-1}"; TREE_ARENA="${CALIB_TREE_ARENA:-1}"
  TREE_PROXY_GRAPH="${CALIB_TREE_PROXY_GRAPH:-1}"
  TREE_WARMUP="${CALIB_TREE_WARMUP:-all}"
  EXIT_REPLICA="${CALIB_EXIT_REPLICA:-1}"; ASYNC_PROXY_SEND="${CALIB_ASYNC_PROXY_SEND:-1}"
  POLICY_ARGS=(--duet_tree_policy "${CALIB_TREE_POLICY:-eagle}"
    --duet_tree_root_count "${ROOTS}" --duet_tree_nv "${NV}"
    --duet_tree_c_tensor "${C_TENSOR}" --duet_tree_fanout_policy backbone
    --duet_tree_proxy_threshold "${PROXY_THRESHOLD}"
    --duet_tree_conf_threshold "${CONF_THRESHOLD}")
else
  TREE_EXEC="${CALIB_TREE_EXEC:-0}"; TREE_ARENA="${CALIB_TREE_ARENA:-0}"
  TREE_PROXY_GRAPH="${CALIB_TREE_PROXY_GRAPH:-0}"; TREE_WARMUP="${CALIB_TREE_WARMUP:-0}"
  EXIT_REPLICA="${CALIB_EXIT_REPLICA:-0}"; ASYNC_PROXY_SEND="${CALIB_ASYNC_PROXY_SEND:-0}"
  POLICY_ARGS=(--duet_tree_policy off)
fi

p1_list_for () {
  local k1="$1" values=() out="" value i
  IFS=',' read -r -a values <<<"${P1_TEMPLATE}"
  local last="${values[$((${#values[@]} - 1))]}"
  for ((i=0; i<=k1; i++)); do
    value="${last}"; if (( i < ${#values[@]} )); then value="${values[$i]}"; fi
    [[ -n "${out}" ]] && out+=","; out+="${value}"
  done
  printf '%s' "${out}"
}

ACTIVE_PGID=""
cleanup () {
  if [[ -n "${ACTIVE_PGID}" ]]; then
    kill -TERM -- "-${ACTIVE_PGID}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${ACTIVE_PGID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

read -r -a CANDIDATES <<<"${K_CANDIDATES}"
read -r -a SEED_ARRAY <<<"${SEEDS}"
run_index=0
for seed_index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$seed_index]}"
  order=("${CANDIDATES[@]}")
  if (( seed_index % 2 == 1 )); then
    order=(); for ((i=${#CANDIDATES[@]}-1; i>=0; i--)); do order+=("${CANDIDATES[$i]}"); done
  fi
  for candidate in "${order[@]}"; do
    IFS=':' read -r k1 k2 <<<"${candidate}"
    [[ "${k1}" =~ ^[0-9]+$ && "${k2}" =~ ^[0-9]+$ ]] || { echo "bad K candidate: ${candidate}" >&2; exit 2; }
    (( k1 >= k2 && k2 >= 1 )) || { echo "require K1>=K2>=1" >&2; exit 2; }
    dir="${OUT}/${MODE}_k1_${k1}_k2_${k2}_seed_${seed}"
    log="${dir}/run.log"
    if [[ "${RESUME:-0}" == 1 ]] && grep -q '^EXIT:0$' "${log}" 2>/dev/null; then
      echo "reuse ${dir}"; continue
    fi
    mkdir -p "${dir}"
    port=$((BASE_PORT + run_index)); run_index=$((run_index + 1))
    p1_list="$(p1_list_for "${k1}")"
    echo "[$(date -Is)] mode=${MODE} K1=${k1} K2=${k2} seed=${seed}"
    set +e
    setsid env SSD_DIST_PORT="${port}" \
      SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0 \
      SSD_TREE_EXEC="${TREE_EXEC}" SSD_TREE_ARENA="${TREE_ARENA}" \
      SSD_TREE_PROXY_GRAPH="${TREE_PROXY_GRAPH}" SSD_TREE_EXEC_WARMUP="${TREE_WARMUP}" \
      SSD_DUET_EXIT_REPLICA="${EXIT_REPLICA}" SSD_ASYNC_PROXY_SEND="${ASYNC_PROXY_SEND}" \
      SSD_PROXY_STREAM=0 timeout --kill-after=30s "${CALIB_TIMEOUT:-60m}" \
      "${PY}" -O bench/bench.py --llama --size 8 \
        --model_path "${MODEL_PATH}" --quant_awq --quant_awq_artifact "${TARGET_AWQ}" \
        --quant_group_size 128 --b 1 --temp "${TEMP}" --input_len "${INPUT_LEN}" --all \
        --max_model_len 2048 --draft_path "${DRAFT_PATH}" --quant_awq_draft \
        --quant_awq_draft_artifact "${DRAFT_AWQ}" --gpus 5 --async --spec --duet \
        --duet_exit_layer "${EXIT_LAYER}" --f 3 --duet_k1 "${k1}" --duet_k2 "${k2}" \
        --duet_p1_fanout "${P1_FANOUT}" --duet_p1_fanout_list "${p1_list}" \
        --duet_p2_budget "${P2_BUDGET}" "${POLICY_ARGS[@]}" \
        --seed "${seed}" --numseqs "${NUMSEQS}" --output_len "${OUTLEN}" \
        "${EXTRA[@]}" >"${log}" 2>&1 &
    leader=$!; ACTIVE_PGID="${leader}"; wait "${leader}"; rc=$?
    kill -TERM -- "-${leader}" 2>/dev/null || true; sleep 1
    kill -KILL -- "-${leader}" 2>/dev/null || true; ACTIVE_PGID=""
    set -e
    echo "EXIT:${rc}" >>"${log}"
    (( rc == 0 )) || { echo "failed rc=${rc}: ${log}" >&2; exit "${rc}"; }
    grep -E "Final Decode Throughput|Avg Tokens per step \(incl recovery\)|Avg Phase [12].*Hit Rate|Avg Phase [12] Accepted Len" \
      "${log}" >"${dir}/metrics.txt" || true
  done
done

"${PY}" tools/duet_calibration/summarize_tps.py "${OUT}" \
  --json-out "${OUT}/summary.json" --strict | tee "${OUT}/summary.txt"
echo "results: ${OUT}/summary.txt"
