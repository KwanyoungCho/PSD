#!/usr/bin/env bash
# P2 candidate-source ablation, chain, temperature 1.0.
#
#   residual = [p_E - p_D]_+   (champion, MESA-SSD §4.2)
#   proxy    = p_E only        (target-distribution-only baseline)
#   draft    = p_D only
#
# Only the P2 candidate source changes; h_i (first-reject distribution) is
# shared by all three arms, so r_i is the single variable.
#
# Measurement rules from docs/duet/00-server-setup.md §9: profiler OFF,
# 3 seeds, arm order rotated per seed.  NOTE: this box shows a run-to-run
# nondeterminism floor (~5% of tokens differ between two identical greedy
# runs), so single-run deltas are not evidence.
set -euo pipefail

ROOT=/home/chokwans99/PSD/ssd
# ONLY_PROXY=1 -> logical K1=0: the draft generates no P1 candidates, so every
# cache root comes from the P2 proxy score.  That makes the three arms compete
# over 100% of the cache instead of the ~4% slice P2 holds when P1 is active.
ONLY_PROXY=${ONLY_PROXY:-0}
_SUFFIX=""; _FLAGS=()
if [ "$ONLY_PROXY" = "1" ]; then _SUFFIX="_onlyproxy"; _FLAGS=(--duet_only_proxy); fi
OUT=${OUT:-$ROOT/experiments/proxy_source_ablation/temp1_3arm_20260908/logs$_SUFFIX}
source "$ROOT/env.sh" > /dev/null

PY="$ROOT/.venv/bin/python"
# TARGET must be the AWQ-CALIBRATED dir, not the original checkpoint: the
# calibration folds the inverse scale into each preceding RMSNorm, so the
# quantized linears only make sense together with the norms stored there.
# Pointing at the original model yields fluent-looking garbage that passes
# every metric check -- always eyeball the generated text.
TARGET=/home/chokwans99/awq_calibrated/layerskip_llama2_70b
DRAFT=/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
ARTIFACT=/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4

GPUS=${GPUS:-2,3,4,5,6}
TEMP=${TEMP:-1.0}
NUMSEQS=${NUMSEQS:-32}          # per dataset; --all => x4.  Pool is 100/dataset.
OUTLEN=${OUTLEN:-256}
SEEDS=${SEEDS:-"42 123 7"}

# §5: profiler and every diagnostic switch OFF -- they distort arm gaps.
export SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0
unset SSD_TREE_STAGE1 SSD_TREE_STAGE2 SSD_TREE_TOPO_TRACE \
      SSD_TREE_NODE_AUDIT SSD_TREE_EXEC_DELAY_MS SSD_TREE_GAP_PROF \
      SSD_CG_INPUT_CHECK SSD_TREE_CALIB_TRACE SSD_DUET_E0_TRACE \
      SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER 2>/dev/null || true
# chain arm: tree executor explicitly off (docs/duet/00 §6)
export SSD_TREE_EXEC=0 SSD_TREE_ARENA=0 SSD_TREE_PROXY_GRAPH=0 SSD_TREE_EXEC_WARMUP=0

mkdir -p "$OUT"
port=18850

for seed in $SEEDS; do
  # §9.3: rotate arm order per seed so GPU drift does not favour one arm.
  case "$seed" in
    42)  arms="residual proxy draft" ;;
    123) arms="proxy draft residual" ;;
    *)   arms="draft residual proxy" ;;
  esac
  for arm in $arms; do
    log="$OUT/${arm}_seed${seed}.log"
    if [ -s "$log" ] && grep -q "EXIT:0" "$log"; then
      echo "[skip] $log"; continue
    fi
    echo "[run] arm=$arm seed=$seed port=$port"
    CUDA_VISIBLE_DEVICES=$GPUS SSD_DIST_PORT=$port \
    "$PY" -O "$ROOT/bench/bench.py" \
      --llama --size 70 --gpus 5 \
      --model_path "$TARGET" --draft_path "$DRAFT" \
      --quant_awq --quant_awq_artifact "$ARTIFACT" \
      --all --numseqs "$NUMSEQS" --output_len "$OUTLEN" \
      --b 1 --temp "$TEMP" --seed "$seed" \
      --async --spec --duet --duet_exit_layer 56 \
      --duet_phase1_k 8 --duet_phase2_k 4 \
      --duet_draft_fan_out 3 --duet_p2_budget 15 \
      --duet_p1_tree_policy off --duet_p2_tree_policy off \
      --duet_proxy_source "$arm" "${_FLAGS[@]+"${_FLAGS[@]}"}" \
      > "$log" 2>&1 && echo "EXIT:0" >> "$log" || echo "EXIT:$?" >> "$log"
    tail -1 "$log"
    port=$((port+1))
  done
done
echo "done -> $OUT"
