#!/usr/bin/env bash
# One short structural decision: confidence tree response cap 6 vs 8.
# This is deliberately not a parameter sweep.  The winner is chosen by
# useful tokens/step together with end-to-end TPS, then frozen for the final
# rotated chain/legacy/confidence comparison.
set -u

ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/confidence_nv_gate"
mkdir -p "${OUT}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1
export SSD_DUET_EXIT_REPLICA=1 SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=0

COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 123
  --input_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1
  --duet_p2_budget 10 --numseqs 2 --output_len 256
  --duet_tree_policy confidence)

cleanup_jobs () {
  local child
  for child in $(jobs -pr); do
    kill "${child}" 2>/dev/null || true
  done
}
trap cleanup_jobs EXIT

run_one () {
  local nv="$1" port="$2" log="${OUT}/confidence_nv${1}.log"
  echo "[$(date -Is)] confidence Nv=${nv}"
  SSD_DIST_PORT="${port}" SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 \
    SSD_PROFILE=0 SSD_PROFILE_DUET=0 \
    timeout 15m "${PY}" -O bench/bench.py "${COMMON[@]}" \
      --duet_tree_nv "${nv}" >"${log}" 2>&1
  local rc=$?
  echo "EXIT:${rc}" >>"${log}"
  grep -E "Final Decode Throughput|Avg Tokens per step|Hit Rate|Accepted Len|Avg target time|Avg target verify|Avg draft step|p2exec stats" \
    "${log}" || true
  return "${rc}"
}

run_one 6 15511
run_one 8 15512
echo CONFIDENCE_NV_GATE_DONE
