#!/usr/bin/env bash
# Collect post-hoc P2-tree labels with thresholds disabled, then recommend
# static proxy/confidence expansion floors.  This is a diagnostic run: trace
# overhead makes its TPS invalid.  Use the generated threshold.env in one
# separate short production A/B before adopting it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/threshold_calibration_${STAMP}}"
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

MODEL_PATH="${MODEL_PATH:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
TARGET_AWQ="${TARGET_AWQ:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_PATH="${DRAFT_PATH:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
DRAFT_AWQ="${DRAFT_AWQ:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

TEMP="${CALIB_TEMP:-0.7}"
INPUT_LEN="${CALIB_INPUT_LEN:-512}"
OUTLEN="${CALIB_OUTLEN:-384}"
NUMSEQS="${CALIB_NUMSEQS:-10}"
SEEDS="${CALIB_SEEDS:-42 123}"
DEPTH="${CALIB_DEPTH:-4}"
K1="${CALIB_K1:-9}"
P1_FANOUT_LIST="${CALIB_P1_FANOUT_LIST:-2,2,2,2,2,2,1,1,1,1}"
ROOTS="${CALIB_ROOTS:-10}"
P2_BUDGET="${CALIB_P2_BUDGET:-10}"
NV="${CALIB_NV:-8}"
C_TENSOR="${CALIB_C_TENSOR:-3}"
RISK_PROFILE="${CALIB_RISK_PROFILE:-balanced}"
BASE_PORT="${CALIB_BASE_PORT:-16600}"
BLOCK_SIZE="${CALIB_BLOCK_SIZE:-256}"
START_BUCKET=$(((INPUT_LEN + K1 + 1 + P2_BUDGET + BLOCK_SIZE - 1) / BLOCK_SIZE))

COMMON=(--llama --size 8
  --model_path "${MODEL_PATH}"
  --quant_awq --quant_awq_artifact "${TARGET_AWQ}"
  --quant_group_size 128 --b 1 --temp "${TEMP}"
  --input_len "${INPUT_LEN}" --all --max_model_len 2048
  --draft_path "${DRAFT_PATH}"
  --quant_awq_draft --quant_awq_draft_artifact "${DRAFT_AWQ}"
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 "${K1}" --duet_k2 "${DEPTH}"
  --duet_p1_fanout 2
  --duet_p1_fanout_list "${P1_FANOUT_LIST}"
  --duet_p2_budget "${P2_BUDGET}")

# Any extra bench arguments are appended before the mandatory calibration
# flags below.  The mandatory flags always win and guarantee an unbiased
# threshold-off EAGLE trace.
EXTRA=("$@")
read -r -a SEED_ARRAY <<<"${SEEDS}"
E0_ARGS=()
CONF_ARGS=()
idx=0
for seed in "${SEED_ARRAY[@]}"; do
  dir="${OUT}/seed_${seed}"
  mkdir -p "${dir}/e0"
  log="${dir}/run.log"
  port=$((BASE_PORT + idx))
  echo "[$(date -Is)] calibration seed=${seed} output=${dir}"
  {
    printf 'PY=%q ' "${PY}"
    printf '%q ' bench/bench.py "${COMMON[@]}" "${EXTRA[@]}"
    printf '%q ' --duet_tree_policy eagle --duet_tree_root_count "${ROOTS}" \
      --duet_tree_nv "${NV}" --duet_tree_c_tensor "${C_TENSOR}" \
      --duet_tree_fanout_policy backbone \
      --duet_tree_proxy_threshold 0 --duet_tree_conf_threshold 0 \
      --seed "${seed}" --numseqs "${NUMSEQS}" --output_len "${OUTLEN}"
    printf '\n'
  } >"${dir}/command.txt"

  SSD_DIST_PORT="${port}" \
  SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0 \
  SSD_DUET_E0_TRACE=1 SSD_DUET_E0_SUBSAMPLE=1 \
  SSD_DUET_E0_DIR="${dir}/e0" \
  SSD_TREE_CALIB_TRACE="${dir}/confidence" \
  SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 SSD_TREE_PROXY_GRAPH=1 \
  SSD_TREE_EXEC_WARMUP="${CALIB_TREE_WARMUP:-all}" \
  SSD_DUET_EXIT_REPLICA=1 \
  SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0 \
    timeout --kill-after=30s "${CALIB_TIMEOUT:-60m}" \
    "${PY}" -O bench/bench.py "${COMMON[@]}" "${EXTRA[@]}" \
      --duet_tree_policy eagle --duet_tree_root_count "${ROOTS}" \
      --duet_tree_nv "${NV}" --duet_tree_c_tensor "${C_TENSOR}" \
      --duet_tree_fanout_policy backbone \
      --duet_tree_proxy_threshold 0 --duet_tree_conf_threshold 0 \
      --seed "${seed}" --numseqs "${NUMSEQS}" \
      --output_len "${OUTLEN}" >"${log}" 2>&1
  echo "EXIT:0" >>"${log}"
  test -s "${dir}/confidence.jsonl"
  test -n "$(find "${dir}/e0" -name 'e0_draft_*.jsonl' -print -quit)"
  E0_ARGS+=("${dir}/e0")
  CONF_ARGS+=("${dir}/confidence.jsonl")
  idx=$((idx + 1))
done

ANALYZER="${ROOT}/tools/duet_calibration/analyze_thresholds.py"
"${PY}" "${ANALYZER}" \
  --e0-dir "${E0_ARGS[@]}" \
  --confidence "${CONF_ARGS[@]}" \
  --depth-cap "${DEPTH}" --risk-profile "${RISK_PROFILE}" \
  --json-out "${OUT}/calibration.json" \
  --config-out "${OUT}/threshold.env" --strict \
  | tee "${OUT}/calibration.txt"

echo "[$(date -Is)] calibration complete"
echo "  report: ${OUT}/calibration.txt"
echo "  JSON:   ${OUT}/calibration.json"
echo "  config: ${OUT}/threshold.env"
