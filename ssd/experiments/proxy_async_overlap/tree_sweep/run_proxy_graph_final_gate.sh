#!/usr/bin/env bash
# Final fixed-shape chain/tree gate after target proxy-graph + warmup work.
# This is one frozen operating point, not a parameter sweep.
set -u

ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${OUT:-${ROOT}/experiments/proxy_async_overlap/tree_sweep/proxy_graph_final_gate}"
mkdir -p "${OUT}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
export SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0
export SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 SSD_TREE_PROXY_GRAPH=1
export SSD_TREE_EXEC_WARMUP=all
export SSD_PROFILE=0 SSD_PROFILE_DUET=0
unset SSD_TREE_ROOT_SHADOW SSD_TREE_NODE_AUDIT SSD_TREE_STAGE1 SSD_TREE_STAGE2

COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7
  --input_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10 --numseqs 10 --output_len 128)

cleanup_run () {
  pkill -9 -u chokwans99 -f \
    "/home/chokwans99/PSD/ssd/bench/bench.py" 2>/dev/null || true
}
trap cleanup_run EXIT

run_one () {
  local arm="$1" seed="$2" port="$3"
  local log="${OUT}/${seed}_${arm}.log"
  local extra=()
  if [[ "${arm}" == "chain" ]]; then
    extra=(--duet_tree_policy off)
  else
    extra=(--duet_tree_policy confidence --duet_tree_nv 8)
  fi
  echo "[$(date -Is)] seed=${seed} arm=${arm}"
  SSD_DIST_PORT="${port}" timeout 12m "${PY}" -O bench/bench.py \
    "${COMMON[@]}" --seed "${seed}" "${extra[@]}" >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  grep -E "Final Decode Throughput|Avg Tokens per step \(incl recovery\)|Avg Phase [12].*Hit Rate|Avg Phase [12] Accepted Len|Avg target time per full step|Avg target verify time|Avg draft step time|p2exec stats" "${log}" || true
  cleanup_run
  sleep 2
  return "${rc}"
}

run_one chain 42 15501
run_one tree 42 15502
run_one tree 123 15503
run_one chain 123 15504
run_one chain 2024 15505
run_one tree 2024 15506

echo PROXY_GRAPH_FINAL_GATE_DONE
