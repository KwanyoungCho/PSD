#!/usr/bin/env bash
# Canonical gate for DUET's phase-specific dynamic trees.  The historical
# filename is retained so existing automation keeps working; public policy
# names are now only P1/P2 off|on.
#
# Order is intentional: one tiny correctness smoke, one final three-seed
# chain/tree comparison with order rotation, then one short timeline per arm.
# This is not a parameter sweep.
set -uo pipefail

# Resolve the checkout instead of pinning eslab18's path.  The same gate can
# then run from the isolated eslab17 worktree without editing the script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/eagle_global_20260807}"
mkdir -p "${OUT}"
cd "${ROOT}"
# bench/bench.py makes its own directory sys.path[0].  Put this checkout
# first so an editable install from another worktree cannot shadow it.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1

MODEL_PATH="${MODEL_PATH:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
TARGET_AWQ="${TARGET_AWQ:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_PATH="${DRAFT_PATH:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
DRAFT_AWQ="${DRAFT_AWQ:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
unset SSD_TREE_ROOT_SHADOW SSD_TREE_STAGE1 SSD_TREE_STAGE2
unset SSD_TREE_EXEC_DELAY_MS

COMMON=(--llama --size 8
  --model_path "${MODEL_PATH}"
  --quant_awq
  --quant_awq_artifact "${TARGET_AWQ}"
  --quant_group_size 128 --b 1 --temp 0.7
  --input_len 512 --all --max_model_len 2048
  --draft_path "${DRAFT_PATH}"
  --quant_awq_draft
  --quant_awq_draft_artifact "${DRAFT_AWQ}"
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10)

run_one () {
  local tag="$1" arm="$2" seed="$3" ns="$4" outlen="$5" profile="$6" port="$7"
  local dir="${OUT}/${tag}_${arm}_s${seed}"
  local log="${dir}/run.log"
  mkdir -p "${dir}"
  if [[ "${RESUME:-0}" == "1" ]] \
      && grep -q '^EXIT:0$' "${log}" 2>/dev/null \
      && [[ -s "${dir}/metrics.txt" ]]; then
    echo "[$(date -Is)] SKIP completed ${tag} ${arm} seed=${seed}"
    cat "${dir}/metrics.txt"
    return 0
  fi
  local p1_policy=off p2_policy=off
  local extra=(--duet_p1_tree_policy off --duet_p2_tree_policy off)
  local exec=0 arena=0 proxy_graph=0 warmup=0
  # Fair chain/tree comparison: these target/proxy optimizations support both
  # policies and therefore must not be enabled only for the tree arm.
  local replica=1 async_send=1
  if [[ "${arm}" == "tree" ]]; then
    p1_policy="${P1_TREE_POLICY:-on}"
    p2_policy="${P2_TREE_POLICY:-on}"
    extra=(--duet_p1_tree_policy "${p1_policy}"
           --duet_p2_tree_policy "${p2_policy}"
           --duet_p1_roots_per_position "${P1_ROOTS_PER_POSITION:-2}"
           --duet_p1_tree_max_nodes "${P1_TREE_MAX_NODES:-13}"
           --duet_p2_tree_max_nodes "${P2_TREE_MAX_NODES:-8}"
           --duet_tree_c_tensor 3
           --duet_tree_fanout_policy backbone
           --duet_tree_proxy_threshold "${TREE_PROXY_THRESHOLD:-0.01}"
           --duet_tree_conf_threshold "${TREE_CONF_THRESHOLD:-0.03}")
    if [[ "${p2_policy}" == "on" ]]; then
      extra+=(--duet_tree_root_count 10)
    fi
    exec=1
    arena=1
    proxy_graph=1
    warmup="${TREE_WARMUP:-all}"
  fi
  echo "[$(date -Is)] START ${tag} ${arm} p1_tree=${p1_policy} "\
"p2_tree=${p2_policy} seed=${seed} "\
"shared_exit_replica=${replica} shared_async_send=${async_send} "\
"tree_exec=${exec} tree_proxy_graph=${proxy_graph} warmup=${warmup}" \
    | tee "${dir}/status.txt"
  SSD_DIST_PORT="${port}" \
  SSD_PROFILE=0 SSD_PROFILE_DUET="${profile}" SSD_PROFILE_DUET_DETAIL=0 \
  SSD_PROFILE_DUET_MAX_EVENTS=12000 SSD_PROFILE_DIR="${dir}" \
  SSD_TREE_EXEC="${exec}" SSD_TREE_ARENA="${arena}" \
  SSD_CHAIN_PROXY_GRAPH=1 SSD_TREE_PROXY_GRAPH="${proxy_graph}" \
  SSD_TREE_EXEC_WARMUP="${warmup}" \
  SSD_DUET_EXIT_REPLICA="${replica}" SSD_ASYNC_PROXY_SEND="${async_send}" \
  SSD_PROXY_STREAM=0 \
    timeout --kill-after=30s 45m \
    "${PY}" -O bench/bench.py "${COMMON[@]}" \
      --seed "${seed}" --numseqs "${ns}" --output_len "${outlen}" \
      "${extra[@]}" >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  echo "[$(date -Is)] END ${tag} ${arm} rc=${rc}" | tee -a "${dir}/status.txt"
  grep -E "Final Decode Throughput|Avg Tokens per step \(incl recovery\)|Avg Phase [12].*Hit Rate|Avg Phase [12] Accepted Len|Avg target time per full step|Avg target verify time|Avg draft step time|p[12]exec stats" "${log}" >"${dir}/metrics.txt" || true
  cat "${dir}/metrics.txt"
  return "${rc}"
}

# 1) Five-minute-class correctness smoke.  Topology tracing is deliberately
# confined to this run because it adds D2H and file I/O.  SMOKE_AUDIT=0 and
# RUN_SCOPE=smoke are diagnostic knobs for reproducing the exact asynchronous
# production path without launching the long gate.
if [[ "${SMOKE_AUDIT:-1}" == "1" ]]; then
  export SSD_TREE_TOPO_TRACE="${OUT}/smoke_topology"
  export SSD_TREE_NODE_AUDIT="${OUT}/smoke_nodes"
fi
run_one smoke tree "${SMOKE_SEED:-42}" "${SMOKE_NS:-2}" \
  "${SMOKE_OUTLEN:-96}" 0 16201 || exit $?
unset SSD_TREE_TOPO_TRACE SSD_TREE_NODE_AUDIT
if [[ "${RUN_SCOPE:-all}" == "smoke" ]]; then
  echo "[$(date -Is)] SMOKE_DONE" | tee "${OUT}/DONE"
  exit 0
fi
if [[ "${RUN_SCOPE:-all}" == "adaptive_fast" ]]; then
  # Reuse the already completed long chain baselines for seeds 42/123.
  # Only the corrected tree policy needs another long run.
  run_one final tree 42 20 384 0 16212 || exit $?
  run_one final tree 123 20 384 0 16213 || exit $?
  run_one profile chain 42 3 256 1 16221 || exit $?
  "${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_chain_s42" --causality-shift \
    >"${OUT}/profile_chain_s42/plot.log" 2>&1
  run_one profile tree 42 3 256 1 16222 || exit $?
  "${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_tree_s42" --causality-shift \
    >"${OUT}/profile_tree_s42/plot.log" 2>&1
  echo "[$(date -Is)] ALL_DONE" | tee "${OUT}/DONE"
  exit 0
fi
if [[ "${RUN_SCOPE:-all}" == "one_tree_profile" ]]; then
  run_one final tree 42 20 384 0 16212 || exit $?
  run_one profile chain 42 3 256 1 16221 || exit $?
  "${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_chain_s42" --causality-shift \
    >"${OUT}/profile_chain_s42/plot.log" 2>&1
  run_one profile tree 42 3 256 1 16222 || exit $?
  "${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_tree_s42" --causality-shift \
    >"${OUT}/profile_tree_s42/plot.log" 2>&1
  echo "[$(date -Is)] ALL_DONE" | tee "${OUT}/DONE"
  exit 0
fi
if [[ "${RUN_SCOPE:-all}" == "profile_only" ]]; then
  run_one profile chain 42 3 256 1 16221 || exit $?
  "${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_chain_s42" --causality-shift \
    >"${OUT}/profile_chain_s42/plot.log" 2>&1
  run_one profile tree 42 3 256 1 16222 || exit $?
  "${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_tree_s42" --causality-shift \
    >"${OUT}/profile_tree_s42/plot.log" 2>&1
  echo "[$(date -Is)] ALL_DONE" | tee "${OUT}/DONE"
  exit 0
fi

# 2) Final distribution gate.  Three seeds, order rotated, no profiler.
run_one final chain 42 20 384 0 16211 || exit $?
run_one final tree 42 20 384 0 16212 || exit $?
run_one final tree 123 20 384 0 16213 || exit $?
run_one final chain 123 20 384 0 16214 || exit $?
run_one final chain 2024 20 384 0 16215 || exit $?
run_one final tree 2024 20 384 0 16216 || exit $?

# 3) Timeline only after correctness/quality passes.  The event cap prevents
# the old ~23k-event profiler stall from contaminating a random span.
run_one profile chain 42 3 256 1 16221 || exit $?
"${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_chain_s42" --causality-shift \
  >"${OUT}/profile_chain_s42/plot.log" 2>&1
run_one profile tree 42 3 256 1 16222 || exit $?
"${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_tree_s42" --causality-shift \
  >"${OUT}/profile_tree_s42/plot.log" 2>&1

echo "[$(date -Is)] ALL_DONE" | tee "${OUT}/DONE"
