#!/usr/bin/env bash
# Produce aligned timelines for every P1/P2 chain/tree combination using the
# current champion configuration.  These are diagnostic profile runs, not TPS
# measurements: DETAIL=1 intentionally records the proxy sub-spans.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/timeline_phase_matrix_champion_20260808}"
GPU_SET="${GPU_SET:-0,1,2,3,4}"
RUN_NS="${RUN_NS:-1}"
RUN_OUTLEN="${RUN_OUTLEN:-256}"
SEED="${SEED:-42}"
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

# Clear every diagnostic that changes the token path or adds unrelated host
# synchronization.  The timeline profiler itself is the only instrumentation.
unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
unset SSD_TREE_ROOT_SHADOW SSD_TREE_NODE_AUDIT SSD_TREE_STAGE1 SSD_TREE_STAGE2
unset SSD_TREE_TOPO_TRACE SSD_TREE_ALLOC_CHECK SSD_TREE_EXEC_DELAY_MS
unset SSD_TREE_EXEC_EAGER_DIAG SSD_TREE_EXEC_CHECK_PCELL_DIAG

COMMON=(--llama --size 8
  --model_path "${MODEL_PATH}"
  --quant_awq --quant_awq_artifact "${TARGET_AWQ}"
  --quant_group_size 128 --b 1 --temp 0.7
  --input_len 512 --all --max_model_len 2048
  --draft_path "${DRAFT_PATH}"
  --quant_awq_draft --quant_awq_draft_artifact "${DRAFT_AWQ}"
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10
  --duet_p1_roots_per_position 2
  --duet_p1_tree_forward_scale 1.0
  --duet_p1_tree_max_nodes 18 --duet_p1_tree_verify_nodes 14
  --duet_p2_tree_max_nodes 8 --duet_p2_tree_verify_nodes 8
  --duet_tree_root_count 10 --duet_tree_c_tensor 3
  --duet_tree_fanout_policy backbone
  --duet_tree_proxy_threshold 0.01 --duet_tree_conf_threshold 0.03
  --duet_p1_tree_start_threshold 0.0 --duet_p1_tree_conf_threshold 0.0)

run_arm () {
  local label="$1" p1="$2" p2="$3" port="$4"
  local dir="${OUT}/${label}"
  local log="${dir}/run.log"
  mkdir -p "${dir}"

  if [[ "${RESUME:-0}" == "1" ]] \
      && grep -q '^EXIT:0$' "${log}" 2>/dev/null \
      && compgen -G "${dir}/duet_profile_target_rank0_*.json" >/dev/null \
      && compgen -G "${dir}/duet_profile_draft_*.json" >/dev/null; then
    echo "[$(date -Is)] SKIP ${label}"
  else
    local exec=0 arena=0 proxy_graph=0 warmup=0
    if [[ "${p1}" == "on" || "${p2}" == "on" ]]; then
      arena=1
      proxy_graph=1
      warmup=all
    fi
    if [[ "${p2}" == "on" ]]; then
      exec=1
    fi

    {
      echo "label=${label}"
      echo "p1_policy=${p1}"
      echo "p2_policy=${p2}"
      echo "git_commit=$(git rev-parse HEAD)"
      echo "seed=${SEED}"
      echo "numseqs_per_dataset=${RUN_NS}"
      echo "output_len=${RUN_OUTLEN}"
      echo "k1=9"
      echo "k2=4"
      echo "p1_nodes=18/14(gen/verify)"
      echo "p2_nodes=8/8(gen/verify)"
      echo "profile_detail=1"
    } >"${dir}/run_meta.env"

    echo "[$(date -Is)] START ${label} P1=${p1} P2=${p2}" \
      | tee "${dir}/status.txt"
    SSD_DIST_PORT="${port}" \
    SSD_PROFILE=0 SSD_PROFILE_DUET=1 SSD_PROFILE_DUET_DETAIL=1 \
    SSD_PROFILE_DUET_MAX_EVENTS=12000 SSD_PROFILE_DIR="${dir}" \
    SSD_TREE_EXEC="${exec}" SSD_TREE_ARENA="${arena}" \
    SSD_CHAIN_PROXY_GRAPH=1 SSD_TREE_PROXY_GRAPH="${proxy_graph}" \
    SSD_TREE_EXEC_WARMUP="${warmup}" \
    SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0 \
      timeout --kill-after=30s 45m \
      "${PY}" -O bench/bench.py "${COMMON[@]}" \
        --seed "${SEED}" --numseqs "${RUN_NS}" \
        --output_len "${RUN_OUTLEN}" \
        --duet_p1_tree_policy "${p1}" \
        --duet_p2_tree_policy "${p2}" >"${log}" 2>&1
    local rc=$?
    echo "EXIT:${rc}" >>"${log}"
    echo "[$(date -Is)] END ${label} rc=${rc}" | tee -a "${dir}/status.txt"
    if ((rc != 0)); then
      return "${rc}"
    fi
  fi

  grep -E "Final Decode Throughput|Avg Tokens per step \(incl recovery\)|Avg Cache Hits|Avg Phase [12].*Hit Rate|Avg Phase [12] Accepted Len|Avg target time per full step|Avg target verify time|Avg draft step time|p[12]exec stats" \
    "${log}" >"${dir}/metrics.txt" || true
  "${PY}" bench/plot_duet_aligned_timeline.py "${dir}" --causality-shift \
    >"${dir}/plot.log" 2>&1
  "${PY}" tools/duet_timeline/summarize_proxy.py "${dir}" \
    >"${dir}/proxy_summary.txt" 2>&1 || true
  find "${dir}" -maxdepth 1 -name 'timeline_*.png' -printf '%f\n' \
    | sort >"${dir}/images.txt"
  cat "${dir}/metrics.txt"
  cat "${dir}/images.txt"
}

run_arm p1_off_p2_off off off 16881 || exit $?
run_arm p1_on_p2_off  on  off 16882 || exit $?
run_arm p1_off_p2_on  off on  16883 || exit $?
run_arm p1_on_p2_on   on  on  16884 || exit $?

{
  echo "commit=$(git rev-parse HEAD)"
  echo "config=K1=9 K2=4 P1=18/14 P2=8/8 R=10 W=10 C=3"
  for label in p1_off_p2_off p1_on_p2_off p1_off_p2_on p1_on_p2_on; do
    echo
    echo "[${label}]"
    cat "${OUT}/${label}/metrics.txt"
    cat "${OUT}/${label}/images.txt"
  done
} >"${OUT}/SUMMARY.txt"

echo "[$(date -Is)] PHASE_COMBO_TIMELINES_DONE" | tee "${OUT}/DONE"
