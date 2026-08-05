#!/usr/bin/env bash
# 23번 RNG 격리 — 라이브 arena(SSD_TREE_EXEC=0)에서 P2 RNG만 토글:
# default(기본 RNG) vs dedicated(전용 gen). 커널은 양쪽 auto 동일.
# dedicated도 hit 하락하면 RNG가 원인, 아니면 kernel(fa2)이 원인.
set -u
ROOT="/home/chokwans99/PSD/ssd"; PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
REV="$(cd "$ROOT" && git rev-parse --short HEAD)"
OUT="${ROOT}/experiments/proxy_async_overlap/tree_sweep/rng_iso_${REV}"
mkdir -p "${OUT}"; cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
wait_clean () { while true; do g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4|sort -n|tail -1); [ "$g" -lt 1000 ]&&break; sleep 60; done; }
TREE=(--llama --size 8 --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --b 1 --temp 0.7 --input_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b
  --quant_awq_draft --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --gpus 5 --async --spec --duet --duet_exit_layer 56 --f 3 --duet_k1 9 --duet_k2 4
  --duet_p1_fanout 2 --duet_p1_fanout_list 2,2,2,2,2,2,1,1,1,1 --duet_p2_budget 10
  --duet_tree_policy level --duet_tree_nv 8 --duet_tree_beta 0.5)
PORT=14500
run () { local lbl="$1" ded="$2" sd="$3"
  grep -qs "^EXIT:0" "${OUT}/${lbl}.log" && { echo "[skip] ${lbl}"; return; }
  wait_clean; PORT=$((PORT+1)); echo "[$(date -Is)] ${lbl} (dedicated=${ded} seed=${sd})"
  SSD_DIST_PORT=${PORT} SSD_TREE_EXEC=0 SSD_TREE_P2_DEDICATED_RNG=${ded} \
    "${PY}" -O bench/bench.py "${TREE[@]}" --seed "${sd}" --numseqs 15 --output_len 384 \
    > "${OUT}/${lbl}.log" 2>&1
  echo "EXIT:$?" >> "${OUT}/${lbl}.log"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader|sort -u); do [ "$(ps -o user= -p $p 2>/dev/null)" = chokwans99 ]&&kill -9 $p 2>/dev/null; done; sleep 8
}
i=0
for sd in 42 123 7; do
  if [ $((i%2)) -eq 0 ]; then run "s${sd}_def" 0 $sd; run "s${sd}_ded" 1 $sd
  else run "s${sd}_ded" 1 $sd; run "s${sd}_def" 0 $sd; fi
  i=$((i+1))
done
echo "RNG_ISO_DONE"
for f in "${OUT}"/s*.log; do b=$(basename $f .log)
  h=$(grep -m1 "Avg Cache Hits" $f|grep -o "[0-9.]*$"); p1=$(grep -m1 "Phase 1 (draft) Hit" $f|grep -o "[0-9.]*$")
  a2=$(grep -m1 "Phase 2 Accepted Len" $f|grep -o "[0-9.]*$")
  echo "$b hit=$h p1hit=$p1 p2al=$a2"
done
