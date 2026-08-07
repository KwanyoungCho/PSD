#!/usr/bin/env bash
# Fast structural gate for the low-knob confidence tree.  This is not a
# parameter sweep: one short chain / legacy-R6 / confidence-R10 rotation.
set -u

ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/confidence_smoke"
mkdir -p "${OUT}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
# Use the already validated target-side overlap machinery.  The code change
# under test is that tree steps now use it too instead of blocking inline.
export SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1
export SSD_PROXY_STREAM=0

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
  --duet_p2_budget 10 --numseqs 2 --output_len 128)

cleanup_jobs () {
  # Limit cleanup to this benchmark command, not every process owned by the
  # user or every process using a GPU.
  pkill -9 -u chokwans99 -f "/home/chokwans99/PSD/ssd/bench/bench.py" \
    2>/dev/null || true
}
trap cleanup_jobs EXIT

run_one () {
  local label="$1" port="$2" exec_flag="$3"
  shift 3
  local log="${OUT}/${label}.log"
  echo "[$(date -Is)] ${label}"
  SSD_DIST_PORT="${port}" SSD_TREE_EXEC="${exec_flag}" \
    SSD_TREE_ARENA=1 SSD_PROFILE=0 SSD_PROFILE_DUET=0 \
    timeout 12m "${PY}" -O bench/bench.py "${COMMON[@]}" "$@" \
    > "${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >> "${log}"
  grep -E "Final Decode Throughput|Phase 1 .*Hit Rate|Phase 2 .*Hit Rate|Phase 1 Accepted Len|Phase 2 Accepted Len|Avg Tokens per step|Avg draft step time|p2exec stats" "${log}" || true
  cleanup_jobs
  sleep 3
  return "${rc}"
}

run_one chain 15301 0 --duet_tree_policy off
run_one level_r6 15302 1 --duet_tree_policy level --duet_tree_nv 8 \
  --duet_tree_beta 0.5 --duet_tree_root_count 6
run_one confidence 15303 1 --duet_tree_policy confidence

echo CONFIDENCE_SMOKE_DONE
