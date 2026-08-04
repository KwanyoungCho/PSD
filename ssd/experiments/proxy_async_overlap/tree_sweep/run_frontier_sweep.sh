#!/usr/bin/env bash
# W×Nv frontier sweep (리뷰2 우선순위 8) — 수정 코드 rev, eslab18.
# 상대 비교 전용: 측정 창 load 캐비앗 필수 (병렬-서버 정책 2026-08-04:
# 절대수치는 17번 클린박스, 상대 sweep은 18번 허용 — load 기록).
# 제약: nv+1 <= W (#19 TREE_GLUE), nv <= K_max=9. R=6 고정 (단일변수).
set -u
ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
REV="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo norev)"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/frontier_${REV}"
mkdir -p "${OUT}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib SSD_PROFILE_DUET=0

wait_clean_gpu () {
  while true; do
    gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
    if [ "${gmax}" -lt 1000 ]; then break; fi
    echo "[guard] gpu_mem=${gmax}MiB — 60s 대기"; sleep 60
  done
}

DUET=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42 --numseqs 25
  --input_len 512 --output_len 384 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1)

PORT=13990
run_one () {
  local label="$1" budget="$2" nv="$3" rc="$4"
  grep -qs "^EXIT:0" "${OUT}/${label}.log" && { echo "[skip] ${label}"; return; }
  wait_clean_gpu
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === ${label} (port ${PORT}) load=$(cut -d' ' -f1 /proc/loadavg) ==="
  { echo "# rev=${REV} load_start=$(cut -d' ' -f1-3 /proc/loadavg)"; } > "${OUT}/${label}.log"
  SSD_DIST_PORT=${PORT} "${PY}" -O bench/bench.py "${DUET[@]}" \
    --duet_p2_budget "${budget}" --duet_tree_policy level \
    --duet_tree_nv "${nv}" --duet_tree_beta 0.5 \
    --duet_tree_root_count "${rc}" \
    >> "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  echo "# load_end=$(cut -d' ' -f1-3 /proc/loadavg)" >> "${OUT}/${label}.log"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 10
}

# 창-기준 체인 앵커 (같은 창 상대화용 — tree 인자 없는 별도 블록)
grep -qs "^EXIT:0" "${OUT}/chain_anchor.log" || {
  wait_clean_gpu
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === chain_anchor (port ${PORT}) ==="
  { echo "# rev=${REV} load_start=$(cut -d' ' -f1-3 /proc/loadavg)"; } > "${OUT}/chain_anchor.log"
  SSD_DIST_PORT=${PORT} "${PY}" -O bench/bench.py "${DUET[@]}" \
    --duet_p2_budget 10 --duet_tree_policy off \
    >> "${OUT}/chain_anchor.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/chain_anchor.log"
  grep -m1 "Final Decode Throughput" "${OUT}/chain_anchor.log" || echo "NO_TPS chain_anchor"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 10
}
# frontier: W×Nv (nv+1<=W; R=min(6, W-?)→R<=W는 config 가드, R=6 고정)
run_one "w6_nv4"  6  4 6
run_one "w8_nv4"  8  4 6
run_one "w8_nv6"  8  6 6
run_one "w10_nv4" 10 4 6
run_one "w10_nv6" 10 6 6
run_one "w10_nv8" 10 8 6
echo "FRONTIER_DONE"
