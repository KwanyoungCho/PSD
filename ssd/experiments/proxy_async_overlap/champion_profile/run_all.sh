#!/usr/bin/env bash
# Champion (E9K24_jit) aligned-timeline profile — per-step breakdown for
# the final config (docs/duet/09 campaign conclusion). ns=20, PROFILE=1.
set -euo pipefail
ROOT="/home/chokwans99/PSD/ssd"
PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/champion_profile"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export SSD_DIST_PORT=12795
export SSD_CUDA_ARCH=8.6
export TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
export SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
label="e9k24_jit_profile"
outdir="${PHASE_DIR}/${label}"
mkdir -p "${outdir}"
echo "[$(date -Is)] === START ${label} ==="
pkill -9 -u chokwans99 -f "bench/bench.py" 2>/dev/null || true; pkill -9 -u chokwans99 -f "multiprocessing.spawn" 2>/dev/null || true
sleep 5
SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 SSD_PROFILE_DUET=1 SSD_PROFILE_DIR="${outdir}" \
  "${PY}" -O bench/bench.py \
  --llama --size 8 \
  --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
  --quant_awq \
  --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4 \
  --quant_group_size 128 \
  --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 20 \
  --input_len 512 --output_len 512 --all --max_model_len 2048 \
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b \
  --quant_awq_draft \
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1 \
  --async --spec \
  --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 4 \
  --duet_draft_fan_out 2 --duet_policy b \
  --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1 \
  > "${outdir}/run.log" 2>&1 || echo "[$(date -Is)] === CRASH ${label} ==="
tps=$(grep "Final Decode Throughput" "${outdir}/run.log" | tail -1 || echo "NO_TPS")
echo "[$(date -Is)] === END ${label}: ${tps} ==="
echo "[$(date -Is)] === PROFILE DONE ==="
