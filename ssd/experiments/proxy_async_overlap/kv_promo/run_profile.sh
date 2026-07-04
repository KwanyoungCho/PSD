#!/usr/bin/env bash
# KV-promo PROFILE: champion + promo ON, PROFILE=1, ns=20. Measures the
# glue-label reduction (legacy ~2.65ms/hit-step -> promo gather+tip).
set -euo pipefail
ROOT="/home/chokwans99/PSD/ssd"; PHASE_DIR="${ROOT}/experiments/proxy_async_overlap/kv_promo"
PY="/home/chokwans99/anaconda3/envs/ssd/bin/python"; cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 SSD_CUDA_ARCH=8.6 TORCH_CUDA_ARCH_LIST=8.6
export SSD_HF_CACHE=/home/chokwans99/.cache/huggingface/hub SSD_DATASET_DIR=/data2/chokwans99/datasets
export MPLCONFIGDIR=/tmp/matplotlib
ARGS=(--llama --size 8 --model_path /data2/chokwans99/awq_calibrated/layerskip_llama2_70b
  --quant_awq --quant_awq_artifact /data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4
  --quant_group_size 128 --gpus 5 --b 1 --temp 0.7 --seed 42 --numseqs 20
  --input_len 512 --output_len 512 --all --max_model_len 2048
  --draft_path /data2/chokwans99/awq_calibrated/tinyllama_1b --quant_awq_draft
  --quant_awq_draft_artifact /data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1
  --async --spec --k 13 --f 3 --duet --duet_exit_layer 56 --duet_phase1_k 9 --duet_phase2_k 4
  --duet_draft_fan_out 2 --duet_policy b --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1)
run_one() { local label="$1" port="$2"; shift 2; local o="${PHASE_DIR}/${label}"; mkdir -p "$o"
  echo "[$(date -Is)] === START ${label} ==="
  pkill -9 -u chokwans99 -f "bench/bench.py" 2>/dev/null||true; pkill -9 -u chokwans99 -f "multiprocessing.spawn" 2>/dev/null||true; sleep 5
  SSD_DIST_PORT=$port SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 SSD_PROFILE_DUET=1 SSD_PROFILE_DIR="$o" "$@" timeout 900 "${PY}" -O bench/bench.py "${ARGS[@]}" > "$o/run.log" 2>&1 || echo "CRASH ${label}"
  echo "[$(date -Is)] === END ${label}: $(grep 'Final Decode' "$o/run.log"|tail -1) ==="
}
run_one "prof_off" 12860
run_one "prof_on"  12861 env SSD_DUET_KV_PROMO=1
echo "[$(date -Is)] === PROF DONE ==="
