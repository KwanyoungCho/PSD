#!/usr/bin/env bash
# 형상 재선정 게이트: tree w10_nv8 (현 챔피언) vs w8_nv6 (frontier
# 스윗스팟) + chain 앵커 — eslab17 인터리브 3-cycle.
set -u
ROOT="$HOME/Parallel_SD/ssd"
PY="/data2/chokwans99/conda_envs/ssd/bin/python"
REV="$(cd "$ROOT" && git rev-parse --short HEAD)"
OUT="$HOME/shape_gate_${REV}"
mkdir -p "${OUT}"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=$HOME/hf_cache CUDA_HOME=$HOME/cuda129
export PATH=$HOME/cuda129/bin:$PATH
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib SSD_PROFILE_DUET=0
wait_clean_box () {
  while true; do
    gmax=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4 | sort -n | tail -1)
    [ "${gmax}" -lt 1000 ] && break
    echo "[guard] gpu_mem=${gmax}MiB"; sleep 60
  done
}
COMMON=(--llama --size 8
  --model_path /data2/chokwans99/awq_calibrated_autoawq/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_ref_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --seed 42 --numseqs 25
  --input_len 512 --output_len 384 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet
  --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4 --duet_p1_fanout 2
  --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1)
PORT=14040
run_one () {
  local label="$1"; shift
  grep -qs "^EXIT:0" "${OUT}/${label}.log" && { echo "[skip] ${label}"; return; }
  wait_clean_box
  PORT=$((PORT + 1))
  echo "[$(date -Is)] === ${label} ==="
  SSD_DIST_PORT=${PORT} "${PY}" -O bench/bench.py "${COMMON[@]}" "$@" \
    > "${OUT}/${label}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${label}.log"
  grep -m1 "Final Decode Throughput" "${OUT}/${label}.log" || echo "NO_TPS ${label}"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "chokwans99" ] && kill -9 $p 2>/dev/null
  done
  sleep 10
}
ARMS=(w10nv8 w8nv6 chain)
arm () {
  local cyc="$1" name="$2"
  case "${name}" in
    chain)  run_one "c${cyc}_chain" --duet_p2_budget 10 --duet_tree_policy off ;;
    w10nv8) run_one "c${cyc}_w10nv8" --duet_p2_budget 10 --duet_tree_policy level \
        --duet_tree_nv 8 --duet_tree_beta 0.5 --duet_tree_root_count 6 ;;
    w8nv6)  run_one "c${cyc}_w8nv6" --duet_p2_budget 8 --duet_tree_policy level \
        --duet_tree_nv 6 --duet_tree_beta 0.5 --duet_tree_root_count 6 ;;
  esac
}
for cyc in 1 2 3; do
  for k in 0 1 2; do
    arm "${cyc}" "${ARMS[$(((k + cyc - 1) % 3))]}"
  done
done
echo "SHAPE_GATE_DONE"
for f in "${OUT}"/c*_*.log; do
  b=$(basename "$f" .log)
  t=$(grep -m1 "Final Decode Throughput" "$f" | grep -o "[0-9.]*" | head -1)
  a=$(grep -m1 "Phase 2 Accepted Len" "$f" | grep -o "[0-9.]*$")
  h=$(grep -m1 "proxy) Hit Rate" "$f" | grep -o "[0-9.]*$")
  ts=$(grep -m1 "Tokens per step on Cache Hit" "$f" | grep -o "[0-9.]*$")
  echo "${b} tps=${t} p2al=${a} p2hit=${h} tokstep=${ts}"
done
