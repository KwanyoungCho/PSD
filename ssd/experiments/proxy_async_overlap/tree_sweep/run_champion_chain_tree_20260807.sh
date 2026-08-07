#!/usr/bin/env bash
# Canonical B=1 DUET champion comparison on the current tree branch.
#
# 1) PROFILE_DUET=1, ns=20/out=512: aligned timelines only.
# 2) PROFILE_DUET=0, ns=50/out=512: paper/champion TPS comparison.
#
# The old 81.91 tok/s headline used the latter workload.  Do not compare it
# with the short ns=10/out=128 tree smoke runs.
set -uo pipefail

ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/champion_chain_tree_20260807_v3}"
mkdir -p "${OUT}/profile_chain" "${OUT}/profile_tree" "${OUT}/perf_chain" "${OUT}/perf_tree"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1

# These were not part of the 81.91 champion.  Keep them out of both arms so
# the chain baseline is the documented one, not a later experimental mix.
unset SSD_DUET_EXIT_REPLICA SSD_ASYNC_PROXY_SEND SSD_PROXY_STREAM
unset SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
unset SSD_TREE_ROOT_SHADOW SSD_TREE_NODE_AUDIT SSD_TREE_STAGE1 SSD_TREE_STAGE2
unset SSD_TREE_TOPO_TRACE SSD_TREE_ALLOC_CHECK SSD_TREE_EXEC_DELAY_MS

COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42
  --input_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10)

run_one () {
  local kind="$1" arm="$2" ns="$3" outlen="$4" port="$5"
  local dir="${OUT}/${kind}_${arm}"
  local log="${dir}/run.log"
  local profile=0 detail=0 timeout_len=45m
  local extra=(--duet_tree_policy off)
  local tree_exec=0 tree_arena=0 tree_proxy_graph=0 tree_warmup=0
  # Shared optimizations stay identical across arms; only the tree executor,
  # tree proxy graph, and tree page warmup differ.
  local exit_replica=1 async_proxy_send=1 proxy_stream=0
  if [[ "${kind}" == "profile" ]]; then
    profile=1
    # Match the historical champion timeline.  DETAIL=1 adds dozens of
    # nested CUDA events and measurably perturbs this tightly overlapped path.
    detail=0
    timeout_len=35m
  fi
  if [[ "${arm}" == "tree" ]]; then
    extra=(--duet_tree_policy eagle --duet_tree_root_count 10
           --duet_tree_nv 8 --duet_tree_c_tensor 3
           --duet_tree_proxy_threshold 0.01
           --duet_tree_conf_threshold 0.03
           --duet_tree_fanout_policy backbone)
    tree_exec=1
    tree_arena=1
    tree_proxy_graph=1
    tree_warmup=all
  fi

  echo "[$(date -Is)] START kind=${kind} arm=${arm} ns=${ns} out=${outlen} "\
"shared_exit_replica=${exit_replica} shared_async_send=${async_proxy_send} "\
"tree_exec=${tree_exec} tree_proxy_graph=${tree_proxy_graph} "\
"warmup=${tree_warmup}" | tee "${dir}/status.txt"
  SSD_DIST_PORT="${port}" \
  SSD_PROFILE=0 SSD_PROFILE_DUET="${profile}" \
  SSD_PROFILE_DUET_DETAIL="${detail}" SSD_PROFILE_DIR="${dir}" \
  SSD_TREE_EXEC="${tree_exec}" SSD_TREE_ARENA="${tree_arena}" \
  SSD_CHAIN_PROXY_GRAPH=1 \
  SSD_TREE_PROXY_GRAPH="${tree_proxy_graph}" \
  SSD_TREE_EXEC_WARMUP="${tree_warmup}" \
  SSD_DUET_EXIT_REPLICA="${exit_replica}" \
  SSD_ASYNC_PROXY_SEND="${async_proxy_send}" SSD_PROXY_STREAM="${proxy_stream}" \
    timeout --kill-after=30s "${timeout_len}" \
    "${PY}" -O bench/bench.py "${COMMON[@]}" \
      --numseqs "${ns}" --output_len "${outlen}" "${extra[@]}" \
      >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  echo "[$(date -Is)] END kind=${kind} arm=${arm} rc=${rc}" | tee -a "${dir}/status.txt"
  grep -E "Final Decode Throughput|Avg Tokens per step \(incl recovery\)|Avg Phase [12].*Hit Rate|Avg Phase [12] Accepted Len|Avg target time per full step|Avg target verify time|Avg draft step time|p2exec stats" "${log}" | tee "${dir}/metrics.txt" || true
  return "${rc}"
}

# Produce the requested pictures first.  These TPS values are diagnostic only.
run_one profile chain 10 512 16101 || true
"${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_chain" --causality-shift \
  >"${OUT}/profile_chain/plot.log" 2>&1 || true
run_one profile tree 10 512 16102 || true
"${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/profile_tree" --causality-shift \
  >"${OUT}/profile_tree/plot.log" 2>&1 || true

# Then answer whether the current chain still reproduces the 80+ champion.
run_one perf chain 50 512 16103 || true
run_one perf tree 50 512 16104 || true

echo "[$(date -Is)] CHAMPION_CHAIN_TREE_DONE" | tee "${OUT}/DONE"
