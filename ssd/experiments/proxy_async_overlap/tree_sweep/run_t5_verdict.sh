#!/usr/bin/env bash
# T5 — 최종 비교 verdict: AR / async-SD(C: k=7 f=6) / DUET-chain(E9K24_jit)
# / DUET-tree(<policy> <budget> <nv> <beta> — T4 champion 인자).
# 5-rep 인터리브 (런-간 편차 상쇄 — 18번 E1 교훈), 오염 가드 포함.
# 사용: bash run_t5_verdict.sh <policy> <budget> <nv> <beta>
set -u
ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/t5_verdict"
mkdir -p "${OUT}"
policy="${1:?policy}"; budget="${2:?budget}"; nv="${3:?nv}"; beta="${4:?beta}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib SSD_PROFILE_DUET=0

wait_clean_box () {
  while true; do
    load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
    if [ "${load}" -lt 24 ] && [ "${gmax}" -lt 1000 ]; then break; fi
    echo "[guard] load=${load} gpu_mem=${gmax}MiB — 60s 대기"; sleep 60
  done
}

COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42 --numseqs 25
  --input_len 512 --output_len 384 --all --max_model_len 2048)
DRAFT=(--draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1)

run_one () {  # label + 나머지 인자
  local label="$1"; shift
  [ -f "${OUT}/${label}.log" ] && { echo "[skip] ${label}"; return; }
  wait_clean_box
  echo "[$(date -Is)] === ${label} ==="
  SSD_DIST_PORT=13940 "${PY}" -O bench/bench.py "${COMMON[@]}" "$@" \
    > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  sleep 10
}

for cyc in 1 2 3 4 5; do
  # AR (TP4만, GPU 4장)
  run_one "c${cyc}_ar" --gpus 4
  # async-SD best C: k=7 f=6
  run_one "c${cyc}_sdC" --gpus 5 "${DRAFT[@]}" --async --spec --k 7 --f 6
  # DUET-chain champion E9K24_jit
  run_one "c${cyc}_chain" --gpus 5 "${DRAFT[@]}" --async --spec --duet \
    --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2 \
    --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 --duet_p2_budget 10 \
    --duet_tree_policy off
  # DUET-tree (T4 champion)
  run_one "c${cyc}_tree" --gpus 5 "${DRAFT[@]}" --async --spec --duet \
    --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2 \
    --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 --duet_p2_budget "${budget}" \
    --duet_tree_policy "${policy}" --duet_tree_nv "${nv}" \
    --duet_tree_beta "${beta}"
done
echo "T5_VERDICT_DONE"
