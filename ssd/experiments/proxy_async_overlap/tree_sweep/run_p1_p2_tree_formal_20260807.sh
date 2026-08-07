#!/usr/bin/env bash
# Formal, profiler-OFF DUET comparison for phase-specific dynamic trees.
#
# This is an ablation gate, not a parameter sweep.  K1/K2, the first P1
# fanout, thresholds, and node budgets stay fixed across all seeds.  Each seed
# runs all four arms on the same server so arm differences are paired against
# that seed/server's chain result.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_formal_20260807}"
SERVER_LABEL="${SERVER_LABEL:-$(hostname -s)}"
GPU_SET="${GPU_SET:-0,1,2,3,4}"
mkdir -p "${OUT}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export SSD_FORCE_SPLIT_K1K2=1
export SSD_DUET_JIT_SHORT=1

MODEL_PATH="${MODEL_PATH:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
TARGET_AWQ="${TARGET_AWQ:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_PATH="${DRAFT_PATH:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
DRAFT_AWQ="${DRAFT_AWQ:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

# Fixed formal-gate configuration.  K1/K2 and the initial P1 fanout match the
# established chain setup.  A depth-K1 path consumes K1 response nodes (the
# cache-key root is not counted), so P1 uses the common K1+K2 wire capacity:
# nine backbone nodes plus four possible alternatives.  P2 retains the
# already validated response cap of eight nodes.
K1="${K1:-9}"
K2="${K2:-4}"
P1_ROOTS_PER_POSITION="${P1_ROOTS_PER_POSITION:-2}"
P1_TREE_MAX_NODES="${P1_TREE_MAX_NODES:-13}"
P2_TREE_MAX_NODES="${P2_TREE_MAX_NODES:-8}"
TREE_PROXY_THRESHOLD="${TREE_PROXY_THRESHOLD:-0.01}"
TREE_CONF_THRESHOLD="${TREE_CONF_THRESHOLD:-0.03}"
RUN_NS="${RUN_NS:-20}"
RUN_OUTLEN="${RUN_OUTLEN:-384}"

# Production path only.  Explicitly clear every diagnostic that can alter the
# token path, synchronize the GPU, or write per-step traces.
unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
unset SSD_TREE_ROOT_SHADOW SSD_TREE_STAGE1 SSD_TREE_STAGE2
unset SSD_TREE_EXEC_DELAY_MS SSD_TREE_TOPO_TRACE SSD_TREE_NODE_AUDIT
unset SSD_TREE_EXEC_EAGER_DIAG SSD_TREE_EXEC_CHECK_PCELL_DIAG

COMMON=(--llama --size 8
  --model_path "${MODEL_PATH}"
  --quant_awq --quant_awq_artifact "${TARGET_AWQ}"
  --quant_group_size 128 --b 1 --temp 0.7
  --input_len 512 --all --max_model_len 2048
  --draft_path "${DRAFT_PATH}"
  --quant_awq_draft --quant_awq_draft_artifact "${DRAFT_AWQ}"
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 "${K1}" --duet_k2 "${K2}"
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10)

arm_order () {
  case "$1" in
    42)   echo "chain p1_tree p2_tree both" ;;
    123)  echo "p1_tree both chain p2_tree" ;;
    2024) echo "p2_tree chain both p1_tree" ;;
    *)    echo "chain p2_tree p1_tree both" ;;
  esac
}

arm_selected () {
  local arm="$1"
  [[ ",${ARMS:-chain,p1_tree,p2_tree,both}," == *",${arm},"* ]]
}

run_one () {
  local arm="$1" seed="$2" ordinal="$3"
  local p1_policy=off p2_policy=off
  case "${arm}" in
    chain) ;;
    p1_tree) p1_policy=on ;;
    p2_tree) p2_policy=on ;;
    both) p1_policy=on; p2_policy=on ;;
    *) echo "unknown arm: ${arm}" >&2; return 2 ;;
  esac

  local dir="${OUT}/${SERVER_LABEL}_${arm}_s${seed}"
  local log="${dir}/run.log"
  mkdir -p "${dir}"
  if [[ "${RESUME:-0}" == "1" ]] \
      && grep -q '^EXIT:0$' "${log}" 2>/dev/null \
      && [[ -s "${dir}/metrics.txt" ]]; then
    echo "[$(date -Is)] SKIP ${SERVER_LABEL} ${arm} seed=${seed}"
    return 0
  fi

  local p2_exec=0 tree_proxy_graph=0 warmup=0
  if [[ "${p1_policy}" == "on" || "${p2_policy}" == "on" ]]; then
    tree_proxy_graph=1
    warmup="${TREE_WARMUP:-all}"
  fi
  if [[ "${p2_policy}" == "on" ]]; then
    p2_exec=1
  fi

  local extra=(
    --duet_p1_tree_policy "${p1_policy}"
    --duet_p2_tree_policy "${p2_policy}"
    --duet_p1_roots_per_position "${P1_ROOTS_PER_POSITION}"
    --duet_p1_tree_max_nodes "${P1_TREE_MAX_NODES}"
    --duet_p2_tree_max_nodes "${P2_TREE_MAX_NODES}"
    --duet_tree_c_tensor 3
    --duet_tree_fanout_policy backbone
    --duet_tree_proxy_threshold "${TREE_PROXY_THRESHOLD}"
    --duet_tree_conf_threshold "${TREE_CONF_THRESHOLD}"
  )
  if [[ "${p2_policy}" == "on" ]]; then
    extra+=(--duet_tree_root_count 10)
  fi

  local port=$((16600 + (seed % 100) * 10 + ordinal))
  {
    echo "server=${SERVER_LABEL}"
    echo "arm=${arm}"
    echo "seed=${seed}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "gpu_set=${GPU_SET}"
    echo "k1=${K1}"
    echo "k2=${K2}"
    echo "p1_policy=${p1_policy}"
    echo "p2_policy=${p2_policy}"
    echo "p1_roots_per_position=${P1_ROOTS_PER_POSITION}"
    echo "p1_tree_max_nodes=${P1_TREE_MAX_NODES}"
    echo "p2_tree_max_nodes=${P2_TREE_MAX_NODES}"
    echo "p1_chain_forward_cells=$((16 * K1))"
    echo "p1_tree_forward_cells_chain_context=$((10 * P1_ROOTS_PER_POSITION * K1))"
    echo "p1_tree_forward_cells_max_tree_context=$(((P1_TREE_MAX_NODES + 1) * P1_ROOTS_PER_POSITION * K1))"
    echo "p2_forward_cells=$((10 * K2))"
    echo "numseqs_per_dataset=${RUN_NS}"
    echo "output_len=${RUN_OUTLEN}"
  } >"${dir}/run_meta.env"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader >"${dir}/gpu_before.csv" 2>&1 || true

  echo "[$(date -Is)] START server=${SERVER_LABEL} arm=${arm} seed=${seed} "\
"P1=${p1_policy} P2=${p2_policy} K1=${K1} K2=${K2} "\
"P1roots=${P1_ROOTS_PER_POSITION} P1nodes=${P1_TREE_MAX_NODES} "\
"P2nodes=${P2_TREE_MAX_NODES}" | tee "${dir}/status.txt"

  SSD_DIST_PORT="${port}" \
  SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0 \
  SSD_TREE_EXEC="${p2_exec}" SSD_TREE_ARENA=1 \
  SSD_CHAIN_PROXY_GRAPH=1 SSD_TREE_PROXY_GRAPH="${tree_proxy_graph}" \
  SSD_TREE_EXEC_WARMUP="${warmup}" \
  SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0 \
    timeout --kill-after=30s 60m \
    "${PY}" -O bench/bench.py "${COMMON[@]}" \
      --seed "${seed}" --numseqs "${RUN_NS}" \
      --output_len "${RUN_OUTLEN}" "${extra[@]}" >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  echo "[$(date -Is)] END server=${SERVER_LABEL} arm=${arm} seed=${seed} rc=${rc}" \
    | tee -a "${dir}/status.txt"
  grep -E "Final Decode Throughput|Avg Tokens per step \(incl recovery\)|Avg Cache Hits|Avg Phase [12].*Hit Rate|Avg Phase [12] Accepted Len|Avg Phase [12] Acceptance Ratio|Avg Tokens per step on Cache (Hit|Miss)|Avg target time per full step|Avg target verify time|Avg draft step time|p[12]exec stats" \
    "${log}" >"${dir}/metrics.txt" || true
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader >"${dir}/gpu_after.csv" 2>&1 || true
  cat "${dir}/metrics.txt"
  return "${rc}"
}

IFS=',' read -r -a seed_list <<<"${SEEDS:-42,123,2024}"
ordinal=0
for seed in "${seed_list[@]}"; do
  for arm in $(arm_order "${seed}"); do
    if ! arm_selected "${arm}"; then
      continue
    fi
    ordinal=$((ordinal + 1))
    run_one "${arm}" "${seed}" "${ordinal}" || exit $?
  done
done

echo "[$(date -Is)] ALL_DONE server=${SERVER_LABEL}" | tee "${OUT}/DONE_${SERVER_LABEL}"
