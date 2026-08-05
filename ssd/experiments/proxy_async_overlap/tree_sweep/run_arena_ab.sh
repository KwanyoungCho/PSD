#!/usr/bin/env bash
# arena vs CPU rollout — 동일-커밋 교대 3×2 (리뷰6 §4: 서로 다른
# 커밋·시간대 단일런 비교 금지). eslab18 로컬, 4×192 + 프로파일.
set -u
ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
REV="$(cd "$ROOT" && git rev-parse --short HEAD)"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/arena_ab_${REV}"
mkdir -p "${OUT}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib SSD_PROFILE_DUET=1

PORT=13990
run_one () {
  local label="$1" arena="$2"
  grep -qs "^EXIT:0" "${OUT}/${label}.log" && { echo "[skip] ${label}"; return; }
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === ${label} (arena=${arena}) ==="
  SSD_DIST_PORT=${PORT} SSD_TREE_ARENA=${arena} \
    "${PY}" -O bench/bench.py --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
    --quant_group_size 128 --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 4 \
    --input_len 512 --output_len 192 --all --max_model_len 2048 \
    --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
    --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
    --async --spec --duet --duet_exit_layer 56 --f 3 \
    --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2 \
    --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 --duet_p2_budget 10 \
    --duet_tree_policy level --duet_tree_nv 8 --duet_tree_beta 0.5 \
    --duet_tree_root_count 6 \
    > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  # 프로파일 json을 라벨명으로 수거
  J=$(grep -m1 -o "/tmp/duet_profile_draft_[0-9]*.json" "${OUT}/${label}.log")
  [ -n "$J" ] && cp "$J" "${OUT}/${label}_draft.json"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 8
}

for cyc in 1 2 3; do
  run_one "cpu${cyc}" 0
  run_one "ar${cyc}" 1
done
echo "ARENA_AB_DONE"
for f in "${OUT}"/*_draft.json; do
  echo "--- $(basename $f)"
  "${PY}" "${ROOT}/experiments/proxy_async_overlap/tree_sweep/p2_span_agg.py" "$f" | head -8
done
