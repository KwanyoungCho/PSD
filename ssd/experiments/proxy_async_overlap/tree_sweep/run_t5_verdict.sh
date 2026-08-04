#!/usr/bin/env bash
# T5 — 최종 비교 verdict: AR / async-SD(C: k=7 f=6) / DUET-chain(E9K24_jit)
# / DUET-tree(<policy> <budget> <nv> <beta> — T4 champion 인자).
# 3-cycle 인터리브 (런-간 편차 상쇄 — 18번 E1 교훈), 오염 가드 포함.
# 리뷰2 수용: skip은 EXIT:0(성공)일 때만 — 크래시 로그는 자동 재실행.
# 사용: bash run_t5_verdict.sh <policy> <budget> <nv> <beta>
set -u
ROOT="$HOME/Parallel_SD/ssd"
PY="/data2/chokwans99/conda_envs/ssd/bin/python"
OUT="$HOME/t5_verdict"
mkdir -p "${OUT}"
policy="${1:?policy}"; budget="${2:?budget}"; nv="${3:?nv}"; beta="${4:?beta}"; rc="${5:-6}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=$HOME/hf_cache
export CUDA_HOME=$HOME/cuda129
export PATH=$HOME/cuda129/bin:$PATH
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib SSD_PROFILE_DUET=0

wait_clean_box () {
  # 사용자 지시 (2026-08-04): GPU 유휴만 확인 — CPU load 조건 제거
  # (타 사용자 CPU 작업 무관하게 GPU 비는 대로 시작; 절대 TPS는
  # 오염 주의, sweep은 동일-창 상대 비교 + verdict는 인터리브)
  while true; do
    gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
    if [ "${gmax}" -lt 1000 ]; then break; fi
    echo "[guard] gpu_mem=${gmax}MiB — 60s 대기"; sleep 60
  done
}

# 주의: bench_helpers.get_model_paths는 --model_path를 --draft_path와
# 함께 줄 때만 존중 (없으면 기본 8B HF-cache 조회 → AR 팔 크래시).
DPATH=(--draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b)
COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated_autoawq/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_ref_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42 --numseqs 25
  --input_len 512 --output_len 384 --all --max_model_len 2048)
DRAFT=("${DPATH[@]}"
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1)

run_one () {  # label + 나머지 인자
  local label="$1"; shift
  grep -qs "^EXIT:0" "${OUT}/${label}.log" && { echo "[skip] ${label}"; return; }
  wait_clean_box
  echo "[$(date -Is)] === ${label} ==="
  SSD_DIST_PORT=13940 "${PY}" -O bench/bench.py "${COMMON[@]}" "$@" \
    > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 10
}

for cyc in 1 2 3; do
  # AR (TP4만, GPU 4장; --draft_path는 경로 해석용 — AR에선 draft 미사용)
  run_one "c${cyc}_ar" --gpus 4 "${DPATH[@]}"
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
    --duet_tree_beta "${beta}" --duet_tree_root_count "${rc}"
done
echo "T5_VERDICT_DONE"
