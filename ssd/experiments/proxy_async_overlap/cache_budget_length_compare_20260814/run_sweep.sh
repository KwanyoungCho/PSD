#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/eslab/chokwans99/PSD/ssd/experiments/proxy_async_overlap/cache_budget_length_compare_20260814
BASE=/home/eslab/chokwans99/baseline
PY=$BASE/.venv-ssd/bin/python
DATA=$BASE/data/specbench_cache_budget_k10.jsonl
TARGET=facebook/layerskip-llama2-70B
DRAFT=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DUET_ROOT=/home/eslab/chokwans99/PSD/ssd
SSD_ROOT=/home/eslab/chokwans99/ssd
GPU_ORDER=${GPU_ORDER:-0,3,5}
DIST_PORT=${DIST_PORT:-18200}
K_VALUES=${K_VALUES:-"9 8"}
METHODS=${METHODS:-"duet only_proxy geo uniform"}
BUDGETS=${BUDGETS:-"2 3 4 5 6 7 8"}
PLOT_AFTER=${PLOT_AFTER:-1}

"$PY" "$ROOT/prepare.py"

common=(
  --target "$TARGET" --draft "$DRAFT" --gpus 3
  --data "$DATA" --template raw --max_new_tokens 1024
  --max_model_len 4096 --extend-draft-rope --allow-nonpaper-context
  --temp 0.7 --top_p 1.0 --seed 1 --warmup 2
)

fanout_for() {
  local k=$1
  local method=$2
  local budget=$3
  "$PY" - "$ROOT/manifests/k${k}.json" "$method" "$budget" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
method, budget = sys.argv[2], int(sys.argv[3])
cell = next(cell for cell in manifest["cells"]
            if cell["method"] == method
            and cell["avg_position_budget"] == budget)
print(json.dumps(cell["fan_out_list"], separators=(",", ":")))
PY
}

complete() {
  local path=$1
  [[ -f "$path" && $(wc -l < "$path") -eq 35 ]]
}

archive_incomplete() {
  local path=$1
  if [[ -f "$path" ]]; then
    mv "$path" "${path}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
}

run_duet_cell() {
  local k=$1
  local method=$2
  local budget=$3
  local positions=$((k + 1))
  local total=$((positions * budget))
  local out_dir=$ROOT/results/k${k}_seed1
  local out=$out_dir/raw/${method}_b${budget}.jsonl
  local log=$out_dir/logs/${method}_b${budget}.log
  if complete "$out"; then
    echo "[k-compare] skip complete k=$k $method b=$budget"
    return
  fi
  archive_incomplete "$out"
  local args=(
    --k2 "$k" --exit-layer 56 --p1-tree off --p2-tree off
    --p1-fanout 1 --proxy-top-k 90 --out "$out" "${common[@]}"
  )
  if [[ "$method" == duet ]]; then
    args+=(--k1 "$k" --p2-budget $((total - positions)))
  else
    args+=(--only-proxy --k1 0 --p2-budget "$total")
  fi
  echo "[k-compare] start k=$k positions=$positions $method b=$budget total=$total"
  CUDA_VISIBLE_DEVICES=$GPU_ORDER DUET_ROOT=$DUET_ROOT SSD_DIST_PORT=$DIST_PORT \
    timeout 43200 "$PY" -O "$BASE/runners/run_duet.py" \
    "${args[@]}" 2>&1 | tee "$log"
}

run_ssd_cell() {
  local k=$1
  local method=$2
  local budget=$3
  local out_dir=$ROOT/results/k${k}_seed1
  local out=$out_dir/raw/${method}_b${budget}.jsonl
  local log=$out_dir/logs/${method}_b${budget}.log
  if complete "$out"; then
    echo "[k-compare] skip complete k=$k $method b=$budget"
    return
  fi
  archive_incomplete "$out"
  local fanout
  fanout=$(fanout_for "$k" "$method" "$budget")
  echo "[k-compare] start k=$k $method b=$budget fanout=$fanout"
  CUDA_VISIBLE_DEVICES=$GPU_ORDER SSD_ROOT=$SSD_ROOT SSD_DIST_PORT=$DIST_PORT \
    timeout 43200 "$PY" -O "$BASE/runners/run_ssd.py" \
    --mode ssd --k "$k" --fan-out-list "$fanout" --jit \
    --out "$out" "${common[@]}" 2>&1 | tee "$log"
}

for k in $K_VALUES; do
  mkdir -p "$ROOT/results/k${k}_seed1/raw" "$ROOT/results/k${k}_seed1/logs"
  for method in $METHODS; do
    for budget in $BUDGETS; do
      if [[ "$method" == duet || "$method" == only_proxy ]]; then
        run_duet_cell "$k" "$method" "$budget"
      else
        run_ssd_cell "$k" "$method" "$budget"
      fi
    done
  done
  "$PY" "$ROOT/analyze.py" --k "$k"
done

if [[ "$PLOT_AFTER" == 1 ]]; then
  "$PY" "$ROOT/plot.py"
fi
