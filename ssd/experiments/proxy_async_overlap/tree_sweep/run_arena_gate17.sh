#!/usr/bin/env bash
# arena 채택 게이트 — eslab17 클린박스 인터리브 (측정 규율: 로컬
# 스모크는 무크래시까지, 채택 판정은 여기서). arms: 트리 CPU-rollout
# vs 트리 arena (SSD_TREE_ARENA=0/1), 3-cycle 교대, verdict급 25×384,
# PROFILE=0 (ON 시 TPS -2~-5% 오염). 마지막에 4×192 프로파일 쌍 1회.
set -u
ROOT="$HOME/Parallel_SD/ssd"
PY="/data2/chokwans99/conda_envs/ssd/bin/python"
REV="$(cd "$ROOT" && git rev-parse --short HEAD)"
OUT="$HOME/arena_gate_${REV}"
mkdir -p "${OUT}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=$HOME/hf_cache
export CUDA_HOME=$HOME/cuda129
export PATH=$HOME/cuda129/bin:$PATH
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib

wait_clean_box () {
  while true; do
    gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
    if [ "${gmax}" -lt 1000 ]; then break; fi
    echo "[guard] gpu_mem=${gmax}MiB — 60s 대기"; sleep 60
  done
}

TREE=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated_autoawq/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_ref_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42
  --input_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 --duet_p2_budget 10
  --duet_tree_policy level --duet_tree_nv 8 --duet_tree_beta 0.5
  --duet_tree_root_count 6)

PORT=14000
run_one () {
  local label="$1" arena="$2" ns="$3" ol="$4" prof="$5"
  grep -qs "^EXIT:0" "${OUT}/${label}.log" && { echo "[skip] ${label}"; return; }
  wait_clean_box
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === ${label} (arena=${arena}) ==="
  SSD_DIST_PORT=${PORT} SSD_TREE_ARENA=${arena} SSD_PROFILE_DUET=${prof} \
    "${PY}" -O bench/bench.py "${TREE[@]}" --numseqs "${ns}" \
    --output_len "${ol}" > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  J=$(grep -m1 -o "/tmp/duet_profile_draft_[0-9]*.json" "${OUT}/${label}.log")
  [ -n "$J" ] && cp "$J" "${OUT}/${label}_draft.json"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 10
}

for cyc in 1 2 3; do
  if [ $((cyc % 2)) -eq 1 ]; then
    run_one "c${cyc}_cpu" 0 25 384 0
    run_one "c${cyc}_ar" 1 25 384 0
  else
    run_one "c${cyc}_ar" 1 25 384 0
    run_one "c${cyc}_cpu" 0 25 384 0
  fi
done
# 프로파일 쌍 (분해용 — TPS 판정에는 미사용)
run_one "prof_cpu" 0 4 192 1
run_one "prof_ar" 1 4 192 1
echo "ARENA_GATE_DONE"
for f in "${OUT}"/c*_*.log; do
  b=$(basename "$f" .log)
  t=$(grep -m1 "Final Decode Throughput" "$f" | grep -o "[0-9.]*" | head -1)
  a=$(grep -m1 "Phase 2 Accepted Len" "$f" | grep -o "[0-9.]*$")
  h=$(grep -m1 "proxy) Hit Rate" "$f" | grep -o "[0-9.]*$")
  p1=$(grep -m1 "Phase 1 Accepted Len" "$f" | grep -o "[0-9.]*$")
  echo "${b} tps=${t} p2al=${a} p2hit=${h} p1al=${p1}"
done
