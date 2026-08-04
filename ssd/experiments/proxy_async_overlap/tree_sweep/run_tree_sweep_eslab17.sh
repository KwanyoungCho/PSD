#!/usr/bin/env bash
# T4 sweep — eslab17 전용 (docs/duet/20-21). eslab18 대비 차이:
#  - PY = /data2 conda_envs/ssd, ROOT = ~/Parallel_SD/ssd
#  - model_path = awq_calibrated_autoawq (8-shard), artifact = autoawq_ref_tp4
#  - CUDA_HOME = ~/cuda129 (flashinfer JIT용 nvcc 12.9 — 이슈: 기본 nvcc 10.1)
#  - pkill은 자기-매치 없는 패턴, 런별 포트 증가
set -uo pipefail
ROOT="$HOME/Parallel_SD/ssd"
PY="/data2/chokwans99/conda_envs/ssd/bin/python"
OUT="$HOME/tree_sweep_results"
mkdir -p "${OUT}" "$HOME/hf_cache"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE="$HOME/hf_cache" SSD_DATASET_DIR=/data2/chokwans99/datasets
export CUDA_HOME="$HOME/cuda129"
export PATH="$HOME/cuda129/bin:$PATH"
export MPLCONFIGDIR=/tmp/mpl_$USER SSD_PROFILE_DUET=0

PORT=13960
run_one () {
  local label="$1" policy="$2" budget="$3" nv="$4" beta="$5"
  [ -f "${OUT}/${label}.log" ] && { echo "[skip] ${label}"; return; }
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === ${label} (port ${PORT}) ==="
  SSD_DIST_PORT=${PORT} "${PY}" -O bench/bench.py --llama --size 8 \
    --model_path /data2/chokwans99/awq_calibrated_autoawq/layerskip_llama2_70b \
    --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_ref_tp4 \
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
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  # cleanup: spawn 자식은 cmdline에 bench.py가 없다 — GPU PID 기준으로 정리
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 8
}

run_one base_off off 10 8 0.5
# 형상 진단(21번 §4.5) 후 신-그리드: backbone 고정(기본값), budget10/nv8,
# 정책 × β만 — 형제-이득 극대 지점 탐색 (구 C-그리드는 폐기)
for policy in level frontier; do
  for beta in 0.3 0.5 0.8; do
    run_one "bb_${policy}_b${beta}" "${policy}" 10 8 "${beta}"
  done
done
echo "SWEEP_DONE"
