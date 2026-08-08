#!/usr/bin/env bash
# Collect post-hoc P1/P2 dynamic-tree labels with every expansion floor
# disabled, then recommend static thresholds for all four axes:
#   P2 proxy / P2 confidence / P1 start / P1 confidence.
#
# This is a diagnostic run: trace overhead makes its TPS invalid.  Use the
# generated threshold.env in one separate short production A/B before
# adopting it.  Both trees run ON with generation == verification budgets
# (no rerank truncation), so serve/walk labels cover the full generated
# tree; the G/M rerank caps remain a separate axis on top of these floors.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/threshold_calibration_p1p2_${STAMP}}"
if [[ -e "${OUT}" ]] && [[ -n "$(find "${OUT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to append to existing non-empty OUT=${OUT}" >&2
  exit 2
fi
mkdir -p "${OUT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Hardware and data/model locations remain overridable so the same collector
# can be used on eslab17, eslab18, or a future model pair.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1

# Diagnostics that alter the token path or execution semantics must stay off;
# the three label traces below are read-only observers of the production path.
unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
unset SSD_TREE_ROOT_SHADOW SSD_TREE_STAGE1 SSD_TREE_STAGE2
unset SSD_TREE_EXEC_DELAY_MS SSD_TREE_NODE_AUDIT
unset SSD_TREE_EXEC_EAGER_DIAG SSD_TREE_EXEC_CHECK_PCELL_DIAG SSD_TREE_DIAG

MODEL_PATH="${MODEL_PATH:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
TARGET_AWQ="${TARGET_AWQ:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_PATH="${DRAFT_PATH:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
DRAFT_AWQ="${DRAFT_AWQ:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

TEMP="${CALIB_TEMP:-0.7}"
INPUT_LEN="${CALIB_INPUT_LEN:-512}"
OUTLEN="${CALIB_OUTLEN:-384}"
NUMSEQS="${CALIB_NUMSEQS:-10}"
SEEDS="${CALIB_SEEDS:-42 123}"
K1="${CALIB_K1:-9}"
K2="${CALIB_K2:-4}"
P1_ROOTS_PER_POSITION="${CALIB_P1_ROOTS_PER_POSITION:-2}"
P1_TREE_FORWARD_SCALE="${CALIB_P1_TREE_FORWARD_SCALE:-1.0}"
P1_TREE_MAX_NODES="${CALIB_P1_TREE_MAX_NODES:-$((2 * K1))}"
P2_TREE_MAX_NODES="${CALIB_P2_TREE_MAX_NODES:-$((2 * K2))}"
P2_WIDTH="${CALIB_P2_WIDTH:-10}"
P2_ROOT_COUNT="${CALIB_P2_ROOT_COUNT:-10}"
C_TENSOR="${CALIB_C_TENSOR:-3}"
RISK_PROFILE="${CALIB_RISK_PROFILE:-balanced}"
BASE_PORT="${CALIB_BASE_PORT:-16700}"

# Preserve the established fanout schedule ratio at any K1
# (K1=9 reproduces 2,2,2,2,2,2,1,1,1,1).
if [[ -z "${CALIB_P1_FANOUT_LIST:-}" ]]; then
  _fan_n=$((K1 + 1))
  _fan_twos=$(((3 * _fan_n + 4) / 5))
  CALIB_P1_FANOUT_LIST=""
  for ((_i = 0; _i < _fan_n; _i++)); do
    _v=1
    if ((_i < _fan_twos)); then _v=2; fi
    CALIB_P1_FANOUT_LIST+="${CALIB_P1_FANOUT_LIST:+,}${_v}"
  done
fi

COMMON=(--llama --size 8
  --model_path "${MODEL_PATH}"
  --quant_awq --quant_awq_artifact "${TARGET_AWQ}"
  --quant_group_size 128 --b 1 --temp "${TEMP}"
  --input_len "${INPUT_LEN}" --all --max_model_len 2048
  --draft_path "${DRAFT_PATH}"
  --quant_awq_draft --quant_awq_draft_artifact "${DRAFT_AWQ}"
  --gpus 5 --async --spec --k "$((K1 + K2))" --duet
  --duet_exit_layer 56 --f 3 --duet_k1 "${K1}" --duet_k2 "${K2}"
  --duet_p1_fanout 2
  --duet_p1_fanout_list "${CALIB_P1_FANOUT_LIST}"
  --duet_p2_budget "${P2_WIDTH}")

# Mandatory calibration flags: both trees ON, every floor at zero, and
# verification budgets equal to generation budgets.  Extra caller arguments
# are appended before these so the unbiased-collection flags always win.
TREE=(--duet_p1_tree_policy on --duet_p2_tree_policy on
  --duet_p1_roots_per_position "${P1_ROOTS_PER_POSITION}"
  --duet_p1_tree_forward_scale "${P1_TREE_FORWARD_SCALE}"
  --duet_p1_tree_max_nodes "${P1_TREE_MAX_NODES}"
  --duet_p2_tree_max_nodes "${P2_TREE_MAX_NODES}"
  --duet_p1_tree_verify_nodes "${P1_TREE_MAX_NODES}"
  --duet_p2_tree_verify_nodes "${P2_TREE_MAX_NODES}"
  --duet_tree_root_count "${P2_ROOT_COUNT}"
  --duet_tree_c_tensor "${C_TENSOR}"
  --duet_tree_fanout_policy backbone
  --duet_tree_proxy_threshold 0 --duet_tree_conf_threshold 0
  --duet_p1_tree_start_threshold 0 --duet_p1_tree_conf_threshold 0)

EXTRA=("$@")
read -r -a SEED_ARRAY <<<"${SEEDS}"
E0_ARGS=()
CONF_ARGS=()
TOPO_ARGS=()
idx=0
for seed in "${SEED_ARRAY[@]}"; do
  dir="${OUT}/seed_${seed}"
  mkdir -p "${dir}/e0"
  log="${dir}/run.log"
  port=$((BASE_PORT + idx))
  echo "[$(date -Is)] calibration seed=${seed} output=${dir}"
  {
    echo "# git_commit=$(git rev-parse HEAD)"
    echo "# server=$(hostname -s) load=$(cat /proc/loadavg)"
    printf 'PY=%q ' "${PY}"
    printf '%q ' bench/bench.py "${COMMON[@]}" "${EXTRA[@]}" "${TREE[@]}" \
      --seed "${seed}" --numseqs "${NUMSEQS}" --output_len "${OUTLEN}"
    printf '\n'
  } >"${dir}/command.txt"

  SSD_DIST_PORT="${port}" \
  SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0 \
  SSD_DUET_E0_TRACE=1 SSD_DUET_E0_SUBSAMPLE=1 \
  SSD_DUET_E0_DIR="${dir}/e0" \
  SSD_TREE_CALIB_TRACE="${dir}/confidence" \
  SSD_TREE_TOPO_TRACE="${dir}/topo" \
  SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 \
  SSD_CHAIN_PROXY_GRAPH=1 SSD_TREE_PROXY_GRAPH=1 \
  SSD_TREE_EXEC_WARMUP="${CALIB_TREE_WARMUP:-all}" \
  SSD_DUET_EXIT_REPLICA=1 \
  SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0 \
    timeout --kill-after=30s "${CALIB_TIMEOUT:-60m}" \
    "${PY}" -O bench/bench.py "${COMMON[@]}" "${EXTRA[@]}" "${TREE[@]}" \
      --seed "${seed}" --numseqs "${NUMSEQS}" \
      --output_len "${OUTLEN}" >"${log}" 2>&1
  echo "EXIT:0" >>"${log}"
  test -s "${dir}/confidence.jsonl"
  test -s "${dir}/topo.draft.jsonl"
  test -s "${dir}/topo.serve.jsonl"
  test -s "${dir}/topo.walk.jsonl"
  test -n "$(find "${dir}/e0" -name 'e0_draft_*.jsonl' -print -quit)"
  E0_ARGS+=("${dir}/e0")
  CONF_ARGS+=("${dir}/confidence.jsonl")
  TOPO_ARGS+=("${dir}/topo")
  idx=$((idx + 1))
done

ANALYZER="${ROOT}/tools/duet_calibration/analyze_thresholds.py"
"${PY}" "${ANALYZER}" \
  --e0-dir "${E0_ARGS[@]}" \
  --confidence "${CONF_ARGS[@]}" \
  --topo-prefix "${TOPO_ARGS[@]}" \
  --depth-cap "${K2}" --p1-depth-cap "${K1}" \
  --risk-profile "${RISK_PROFILE}" \
  --json-out "${OUT}/calibration.json" \
  --config-out "${OUT}/threshold.env" --strict \
  | tee "${OUT}/calibration.txt"

echo "[$(date -Is)] calibration complete"
echo "  report: ${OUT}/calibration.txt"
echo "  JSON:   ${OUT}/calibration.json"
echo "  config: ${OUT}/threshold.env"
