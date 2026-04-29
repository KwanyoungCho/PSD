#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <fanout_f>" >&2
  exit 2
fi

F="$1"
K="${K:-7}"
TEMP=0.7
SEED=42
NUMSEQS=32
OUTLEN=512
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="$ROOT/experiments/paper_baselines/ssd_awq_70b_f_sweep_ns32_len512"
OUT="$BASE/20260429_k${K}_f${F}_temp07_seed${SEED}_ns${NUMSEQS}_all"

PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
TARGET="${TARGET:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
DRAFT="${DRAFT:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
TGT_ART="${TGT_ART:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_ART="${DRAFT_ART:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

mkdir -p "$OUT"
if [ -s "$OUT/run.log" ]; then
  echo "run.log already exists: $OUT/run.log" >&2
  echo "move/remove it or choose a new output directory before rerunning." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export SSD_PROFILE_MESA=1
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "[run_one] f=$F k=$K numseqs=$NUMSEQS output_len=$OUTLEN gpus=$CUDA_VISIBLE_DEVICES"
echo "[run_one] out=$OUT"

(
  cd "$ROOT"
  "$PY" -O bench/bench.py \
    --llama --size 8 \
    --model_path "$TARGET" \
    --draft_path "$DRAFT" \
    --quant_awq --quant_awq_artifact "$TGT_ART" --quant_group_size 128 \
    --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
    --async --spec \
    --k "$K" --f "$F" \
    --gpus 5 --b 1 \
    --temp "$TEMP" --seed "$SEED" \
    --numseqs "$NUMSEQS" --output_len "$OUTLEN" --all \
    --max_model_len 2048
) >"$OUT/run.log" 2>&1

target_profile="$(grep -oE '/tmp/mesa_profile_target_rank0_[0-9]+\.json' "$OUT/run.log" | tail -1 || true)"
draft_profile="$(grep -oE '/tmp/mesa_profile_draft_[0-9]+\.json' "$OUT/run.log" | tail -1 || true)"
if [ -n "$target_profile" ] && [ -f "$target_profile" ]; then
  cp "$target_profile" "$OUT/"
fi
if [ -n "$draft_profile" ] && [ -f "$draft_profile" ]; then
  cp "$draft_profile" "$OUT/"
fi

(
  cd "$ROOT"
  "$PY" bench/summarize_ssd_run.py "$OUT" --k "$K" \
    --append-index "$BASE/summary_index.csv"
) >"$OUT/postprocess.log" 2>&1

echo "[run_one] done: $OUT"
