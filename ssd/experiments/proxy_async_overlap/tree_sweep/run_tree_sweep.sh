#!/usr/bin/env bash
# T4 — P2-tree 파라미터 sweep 하네스 (docs/duet/20 로드맵).
# 그리드: policy × R(root 수 = budget) × N_v × β. 각 조합 1런 (탐색),
# 상위 후보만 5-rep 인터리브 verdict (run_tree_verdict.sh — 추후).
#
# 오염 가드 (docs/duet/19 이슈): 각 런 전 CPU load·GPU 점유 확인 —
# 기준 초과 시 대기 (타 사용자 작업과의 공유 박스 교훈).
set -uo pipefail
ROOT="/home/chokwans99/PSD/ssd"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/results"
mkdir -p "${OUT}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
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

run_one () {
  local label="$1" policy="$2" budget="$3" nv="$4" beta="$5"
  [ -f "${OUT}/${label}.log" ] && { echo "[skip] ${label}"; return; }
  wait_clean_box
  pkill -9 -u chokwans99 -f "python -O bench/bench.py" 2>/dev/null || true; sleep 8
  echo "[$(date -Is)] === ${label} ==="
  SSD_DIST_PORT=13920 "${PY}" -O bench/bench.py --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
    --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
    --quant_group_size 128 --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 25 \
    --input_len 512 --output_len 384 --all --max_model_len 2048 \
    --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
    --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
    --async --spec --duet --duet_exit_layer 56 --f 3 \
    --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2 \
    --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 \
    --duet_p2_budget "${budget}" \
    --duet_tree_policy "${policy}" --duet_tree_nv "${nv}" \
    --duet_tree_beta "${beta}" \
    > "${OUT}/${label}.log" 2>&1
  echo "[$(date -Is)] ${label} exit: $?"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS"
  pkill -9 -u chokwans99 -f "python -O bench/bench.py" 2>/dev/null || true
}

# 기준선 (tree off)
run_one base_off off 10 8 0.5
# 그리드 (E1 근거로 압축: R∈{8,10}, N_v∈{6,8}, β∈{0.5,1.0}, 정책 2종)
# 이슈 #19: nv+1 <= budget(W) — R8은 nv6만 유효 (R8+nv8 조합 제거)
for policy in level frontier; do
  for budget in 8 10; do
    for nv in 6 8; do
      if [ $((nv + 1)) -gt "${budget}" ]; then continue; fi
      for beta in 0.5 1.0; do
        run_one "t_${policy}_R${budget}_nv${nv}_b${beta}" \
          "${policy}" "${budget}" "${nv}" "${beta}"
      done
    done
  done
done
echo "SWEEP_DONE"
