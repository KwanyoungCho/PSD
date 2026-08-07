#!/usr/bin/env bash
# One profiled pair after the confidence smoke gate.  Four requests are enough
# to attribute spans; this run is not used for the final quality verdict.
set -u

ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/confidence_timeline_final_20260806"
mkdir -p "${OUT}/chain" "${OUT}/confidence"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
export SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0

COMMON=(--llama --size 4
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
  --duet_p2_budget 10 --numseqs 2 --output_len 192)

cleanup_jobs () {
  pkill -9 -u chokwans99 -f "/home/chokwans99/PSD/ssd/bench/bench.py" \
    2>/dev/null || true
}
trap cleanup_jobs EXIT

run_one () {
  local label="$1" port="$2"
  shift 2
  echo "[$(date -Is)] ${label}"
  SSD_DIST_PORT="${port}" SSD_TREE_EXEC=1 SSD_PROFILE_DUET=1 \
    SSD_PROFILE_DIR="${OUT}/${label}" \
    timeout 12m "${PY}" -O bench/bench.py "${COMMON[@]}" "$@" \
    > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  grep -E "Final Decode Throughput|Phase 1 Accepted Len|Phase 2 Accepted Len|Avg draft step time|p2exec stats" \
    "${OUT}/${label}.log" || true
  cleanup_jobs
  sleep 3
}

run_one chain 15401 --duet_tree_policy off
run_one confidence 15402 --duet_tree_policy confidence

"${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/chain" \
  --causality-shift
"${PY}" bench/plot_duet_aligned_timeline.py "${OUT}/confidence" \
  --causality-shift
echo CONFIDENCE_TIMELINE_DONE
