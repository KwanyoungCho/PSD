#!/usr/bin/env bash
# 5-way parallel comparison on Llama3-8B + Qwama-0.5B (TP=1, 8 GPUs).
#
# GPU allocation:
#   0,1 : async SD k=3 f=3   (PROFILE=1)
#   2,3 : MESA  k1=2 k2=1    (PROFILE=1)  ← breakdown analysis
#   4,5 : MESA  k1=2 k2=1    (PROFILE=0)  ← cold-path TPS
#   6   : SD sync k=3        (PROFILE=1)
#   7   : AR (no spec)       (PROFILE=1)
#
# Same config base as existing qwama_llama3_8b_breakdown_k2:
#   --temp 0.7 --seed 42 --numseqs 50 --input_len 512 --output_len 512
#   --all --max_model_len 2048   (200 prompts across 4 datasets)
#   target = layerskip-llama3-8B (dense fp16)
#   draft  = Qwama-0.5B-Instruct (cross-family Qwen2 arch w/ Llama-3 vocab)

set -uo pipefail

ROOT="/home/chokwans99/PSD/ssd"
BASE="${ROOT}/experiments/qwama_llama3_8b_breakdown_k2"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"
TARGET="/data2/chokwans99/models/layerskip-llama3-8B"
DRAFT="/data2/chokwans99/models/models--turboderp--Qwama-0.5B-Instruct/snapshots/fa73d1257f7681af595c86521b332fe26981f179"

COMMON_ENV=(
  SSD_CUDA_ARCH=8.6
  TORCH_CUDA_ARCH_LIST=8.6
  SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub
  SSD_DATASET_DIR=/data2/chokwans99/datasets
  MPLCONFIGDIR=/tmp/matplotlib
)

COMMON_ARGS=(
  --llama --size 8
  --model_path "${TARGET}"
  --b 1 --temp 0.7 --seed 42 --numseqs 50
  --input_len 512 --output_len 512 --all --max_model_len 2048
)

cd "${ROOT}"
pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# -------------------- 1. async SD k=3 f=3 PROFILE=1 (GPUs 0,1) --------------------
OUT_ASYNC="${BASE}/async_sd_k3_f3_p1"
mkdir -p "${OUT_ASYNC}"
(
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=0,1 SSD_DIST_PORT=12680 \
    SSD_PROFILE_MESA=1 SSD_PROFILE_DIR="${OUT_ASYNC}" SSD_PROFILE_MESA_DETAIL=0 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
      --gpus 2 --draft_path "${DRAFT}" \
      --async --spec --k 3 --f 3 \
    > "${OUT_ASYNC}/run.log" 2>&1
  echo "[$(date -Is)] async_sd_k3_f3_p1 done" >> "${OUT_ASYNC}/run.log"
) &
echo "async_sd_k3_f3_p1   PID=$!  GPUs=0,1"

# -------------------- 2. MESA k1=2 k2=1 PROFILE=1 (GPUs 2,3) --------------------
OUT_MESA_P1="${BASE}/mesa_K1_2_K2_1_dfo2_pfo1_exit18_p1"
mkdir -p "${OUT_MESA_P1}"
(
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=2,3 SSD_DIST_PORT=12682 \
    SSD_FORCE_SPLIT_K1K2=1 \
    SSD_PROFILE_MESA=1 SSD_PROFILE_DIR="${OUT_MESA_P1}" SSD_PROFILE_MESA_DETAIL=0 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
      --gpus 2 --draft_path "${DRAFT}" \
      --async --spec --k 3 --f 3 \
      --mesa --mesa_exit_layer 18 --mesa_phase1_k 2 --mesa_phase2_k 1 \
      --mesa_draft_fan_out 2 --mesa_policy b \
    > "${OUT_MESA_P1}/run.log" 2>&1
  echo "[$(date -Is)] mesa_K1_2_K2_1_p1 done" >> "${OUT_MESA_P1}/run.log"
) &
echo "mesa_K1_2_K2_1_p1   PID=$!  GPUs=2,3"

# -------------------- 3. MESA k1=2 k2=1 PROFILE=0 (GPUs 4,5) --------------------
OUT_MESA_P0="${BASE}/mesa_K1_2_K2_1_dfo2_pfo1_exit18_p0"
mkdir -p "${OUT_MESA_P0}"
(
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=4,5 SSD_DIST_PORT=12684 \
    SSD_FORCE_SPLIT_K1K2=1 \
    SSD_PROFILE_MESA=0 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
      --gpus 2 --draft_path "${DRAFT}" \
      --async --spec --k 3 --f 3 \
      --mesa --mesa_exit_layer 18 --mesa_phase1_k 2 --mesa_phase2_k 1 \
      --mesa_draft_fan_out 2 --mesa_policy b \
    > "${OUT_MESA_P0}/run.log" 2>&1
  echo "[$(date -Is)] mesa_K1_2_K2_1_p0 done" >> "${OUT_MESA_P0}/run.log"
) &
echo "mesa_K1_2_K2_1_p0   PID=$!  GPUs=4,5"

# -------------------- 4. SD sync k=3 PROFILE=1 (GPU 6) --------------------
OUT_SD="${BASE}/sd_k3_p1"
mkdir -p "${OUT_SD}"
(
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=6 SSD_DIST_PORT=12686 \
    SSD_PROFILE_MESA=1 SSD_PROFILE_DIR="${OUT_SD}" SSD_PROFILE_MESA_DETAIL=0 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
      --gpus 1 --draft_path "${DRAFT}" \
      --spec --k 3 \
    > "${OUT_SD}/run.log" 2>&1
  echo "[$(date -Is)] sd_k3_p1 done" >> "${OUT_SD}/run.log"
) &
echo "sd_k3_p1            PID=$!  GPUs=6"

# -------------------- 5. AR PROFILE=1 (GPU 7) --------------------
OUT_AR="${BASE}/ar_p1"
mkdir -p "${OUT_AR}"
(
  env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES=7 SSD_DIST_PORT=12687 \
    SSD_PROFILE_MESA=1 SSD_PROFILE_DIR="${OUT_AR}" SSD_PROFILE_MESA_DETAIL=0 \
    "${PY}" -O bench/bench.py "${COMMON_ARGS[@]}" \
      --gpus 1 \
    > "${OUT_AR}/run.log" 2>&1
  echo "[$(date -Is)] ar_p1 done" >> "${OUT_AR}/run.log"
) &
echo "ar_p1               PID=$!  GPUs=7"

wait
echo ""
echo "[$(date -Is)] === ALL 5 RUNS COMPLETE ==="
for d in "${OUT_ASYNC}" "${OUT_MESA_P1}" "${OUT_MESA_P0}" "${OUT_SD}" "${OUT_AR}"; do
  echo ""
  echo "=== ${d##*/} ==="
  grep -E "Final Decode|Avg target|Avg Cache|Avg Phase|Avg Tokens per step|Avg draft step" "${d}/run.log" 2>/dev/null | grep -v Generation
done
