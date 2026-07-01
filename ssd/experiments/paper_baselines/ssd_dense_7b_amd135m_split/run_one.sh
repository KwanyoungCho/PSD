#!/usr/bin/env bash
# Split-K1/K2 (Policy B) on layerskip-llama2-7B (fp16 dense) + AMD-Llama-135m (fp16 dense).
# Mirrors final_experiments/ours_k1_7_k2_5_dfo2_pfo1_exit56 but for a smaller target/draft
# pair without AWQ. Pass mesa_exit_layer as the single positional arg.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <mesa_exit_layer>" >&2
  exit 2
fi

EXIT_LAYER="$1"
K1="${K1:-7}"
K2="${K2:-5}"
K=$((K1 + K2))
DFO="${DFO:-2}"
PFO="${PFO:-1}"
F=$((DFO + PFO))
TEMP="${TEMP:-0.7}"
SEED="${SEED:-42}"
NUMSEQS="${NUMSEQS:-50}"
INLEN="${INLEN:-512}"
OUTLEN="${OUTLEN:-512}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="$ROOT/experiments/paper_baselines/ssd_dense_7b_amd135m_split"
OUT="$BASE/20260430_split_k${K}_K1_${K1}_K2_${K2}_dfo${DFO}_pfo${PFO}_exit${EXIT_LAYER}_temp07_seed${SEED}_ns${NUMSEQS}_in${INLEN}_out${OUTLEN}_all${OUT_TAG:-}"

PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
TARGET="${TARGET:-/data2/chokwans99/models/layerskip-llama2-7B}"
DRAFT="${DRAFT:-/data2/chokwans99/models/AMD-Llama-135m-fp16}"

mkdir -p "$OUT"
if [ -s "$OUT/run.log" ]; then
  echo "run.log already exists: $OUT/run.log" >&2
  echo "move/remove it or choose a new output directory before rerunning." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export SSD_PROFILE_MESA="${SSD_PROFILE_MESA:-1}"
export SSD_FORCE_SPLIT_K1K2=1
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/data2/chokwans99/models}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export SSD_DIST_PORT="${SSD_DIST_PORT:-12460}"
export SSD_PROFILE_DIR="$OUT"

echo "[run_split_7b] exit=$EXIT_LAYER k=$K K1=$K1 K2=$K2 dfo=$DFO pfo=$PFO f=$F"
echo "[run_split_7b] numseqs=$NUMSEQS input_len=$INLEN output_len=$OUTLEN temp=$TEMP seed=$SEED gpus=$CUDA_VISIBLE_DEVICES"
echo "[run_split_7b] out=$OUT"

(
  cd "$ROOT"
  "$PY" -O bench/bench.py \
    --llama --size 8 \
    --model_path "$TARGET" \
    --draft_path "$DRAFT" \
    --async --spec \
    --k "$K" --f "$F" \
    --mesa --mesa_exit_layer "$EXIT_LAYER" \
    --mesa_phase1_k "$K1" --mesa_phase2_k "$K2" \
    --mesa_draft_fan_out "$DFO" --mesa_policy b \
    --gpus 2 --b 1 \
    --temp "$TEMP" --seed "$SEED" \
    --numseqs "$NUMSEQS" --input_len "$INLEN" --output_len "$OUTLEN" --all \
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

echo "[run_split_7b] done: $OUT"
