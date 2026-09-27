#!/usr/bin/env bash
# All-layer P2 candidate-form probe across datasets x seeds.
#
# Every policy is scored on the SAME steps inside one run (paired), so seeds
# and datasets are here to test generality, not to beat down arm-comparison
# noise.  Metrics: top-R coverage of the engine's true recovery distribution,
# the distributional distance of each candidate FORM to that distribution,
# and the character of p^E relative to p_T and p_D.
set -euo pipefail
ROOT=/home/chokwans99/PSD/ssd
OUT=${OUT:-$ROOT/experiments/proxy_source_ablation/probe_rankprofile_20260910/out}
source "$ROOT/env.sh" > /dev/null
PY="$ROOT/.venv/bin/python"
TARGET=/home/chokwans99/awq_calibrated/layerskip_llama2_70b   # calibrated dir: norms live here
DRAFT=/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
ARTIFACT=/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4

GPUS=${GPUS:-2,3,4,5,6}
NUMSEQS=${NUMSEQS:-32}
OUTLEN=${OUTLEN:-256}
SEEDS=${SEEDS:-"42"}
DATASETS=${DATASETS:-"humaneval alpaca c4 gsm"}

export SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0
export SSD_TREE_EXEC=0 SSD_TREE_ARENA=0 SSD_TREE_PROXY_GRAPH=0 SSD_TREE_EXEC_WARMUP=0
export SSD_DUET_EXIT_REPLICA=1        # probe needs the rank0 lm_head replica
mkdir -p "$OUT"; port=19000

for ds in $DATASETS; do
  case "$ds" in
    humaneval) flag="--humaneval" ;; alpaca) flag="--alpaca" ;;
    c4) flag="--c4" ;; gsm) flag="" ;;          # gsm is bench.py's default
    *) echo "unknown dataset $ds"; exit 1 ;;
  esac
  for seed in $SEEDS; do
    tag="${ds}_seed${seed}"; js="$OUT/$tag.json"
    if [ -s "$js" ]; then echo "[skip] $tag"; port=$((port+1)); continue; fi
    echo "[run] $tag port=$port"
    CUDA_VISIBLE_DEVICES=$GPUS SSD_DIST_PORT=$port \
    SSD_DUET_PROBE_LAYERS=all SSD_DUET_PROBE_TOPM=64 SSD_DUET_PROBE_OUT="$js" \
    "$PY" -O "$ROOT/bench/bench.py" \
      --llama --size 70 --gpus 5 \
      --model_path "$TARGET" --draft_path "$DRAFT" \
      --quant_awq --quant_awq_artifact "$ARTIFACT" \
      $flag --numseqs "$NUMSEQS" --output_len "$OUTLEN" \
      --b 1 --temp 1.0 --seed "$seed" \
      --async --spec --duet --duet_exit_layer 56 \
      --duet_phase1_k 8 --duet_phase2_k 4 \
      --duet_draft_fan_out 3 --duet_p2_budget 15 \
      --duet_p1_tree_policy off --duet_p2_tree_policy off \
      --duet_only_proxy > "$OUT/$tag.log" 2>&1 || true
    # bench.py can exit 0 after an engine exception, so the artefact is the
    # only trustworthy success signal (this is how the c4 runs silently died).
    if [ -s "$js" ]; then echo "  OK"; else
      echo "  FAILED -- $(grep -m1 -E 'Error|ValueError' "$OUT/$tag.log" || echo '?')"
    fi
    port=$((port+1))
  done
done
echo "done -> $OUT"
