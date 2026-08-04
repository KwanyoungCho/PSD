#!/usr/bin/env bash
# 재검증 verdict (리뷰2 수용 후 — 이슈 #27/#28/#31/#33/#34/#35 적용 rev).
# 필수 대조군 (리뷰2): chain-R6 / tree-R6 / chain-R10 / tree-R10.
#  - chain-R6는 chain-budget6로 구현: top-6 P_iv 후보만 서빙 — 상위-후보
#    집중 효과를 topology와 분리 (토큰축 등가; 10폭-4사석 구현 대신
#    코드 불변 해석. 시간축은 CG 폭 차이 有 — P2AL 비교가 목적).
#  - 라벨에 코드 rev 포함 (stale-log 방지 노트 수용), EXIT:0만 skip.
#  - 3-cycle 인터리브, 사이클 내 팔 순서 회전 (고정-순서 편향 노트 수용).
set -u
ROOT="$HOME/Parallel_SD/ssd"
PY="/data2/chokwans99/conda_envs/ssd/bin/python"
REV="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo norev)"
OUT="$HOME/reverdict_${REV}"
mkdir -p "${OUT}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=$HOME/hf_cache
export CUDA_HOME=$HOME/cuda129
export PATH=$HOME/cuda129/bin:$PATH
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib SSD_PROFILE_DUET=0

# 선행 T5 AR 재실행과 충돌 방지 — 종료까지 대기
while pgrep -u chokwans99 -f "run_t5_""verdict" >/dev/null 2>&1; do
  echo "[wait] t5_verdict 진행 중 — 120s"; sleep 120
done

wait_clean_box () {
  while true; do
    gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
    if [ "${gmax}" -lt 1000 ]; then break; fi
    echo "[guard] gpu_mem=${gmax}MiB — 60s 대기"; sleep 60
  done
}

DPATH=(--draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b)
COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated_autoawq/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_ref_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42 --numseqs 25
  --input_len 512 --output_len 384 --all --max_model_len 2048)
DRAFT=("${DPATH[@]}"
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1)
DUET=(--gpus 5 "${DRAFT[@]}" --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1)

PORT=13980
run_one () {
  local label="$1"; shift
  grep -qs "^EXIT:0" "${OUT}/${label}.log" && { echo "[skip] ${label}"; return; }
  wait_clean_box
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === ${label} (port ${PORT}) ==="
  SSD_DIST_PORT=${PORT} "${PY}" -O bench/bench.py "${COMMON[@]}" "$@" \
    > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 10
}

arm () {
  local cyc="$1" name="$2"
  case "${name}" in
    chainR10) run_one "c${cyc}_chainR10" "${DUET[@]}" \
        --duet_p2_budget 10 --duet_tree_policy off ;;
    chainR6)  run_one "c${cyc}_chainR6" "${DUET[@]}" \
        --duet_p2_budget 6 --duet_tree_policy off ;;
    treeR6)   run_one "c${cyc}_treeR6" "${DUET[@]}" \
        --duet_p2_budget 10 --duet_tree_policy level --duet_tree_nv 8 \
        --duet_tree_beta 0.5 --duet_tree_root_count 6 ;;
    treeR10)  run_one "c${cyc}_treeR10" "${DUET[@]}" \
        --duet_p2_budget 10 --duet_tree_policy level --duet_tree_nv 8 \
        --duet_tree_beta 0.5 --duet_tree_root_count 10 ;;
  esac
}

ARMS=(chainR10 chainR6 treeR6 treeR10)
for cyc in 1 2 3; do
  for k in 0 1 2 3; do
    arm "${cyc}" "${ARMS[$(((k + cyc - 1) % 4))]}"   # 사이클별 순서 회전
  done
done
echo "REVERDICT_DONE"
