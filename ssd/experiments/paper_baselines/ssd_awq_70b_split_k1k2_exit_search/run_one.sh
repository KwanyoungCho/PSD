#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <mesa_exit_layer>" >&2
  exit 2
fi

EXIT_LAYER="$1"
K1="${K1:-7}"
K2="${K2:-4}"
K=$((K1 + K2))
DFO="${DFO:-1}"
PFO="${PFO:-2}"
F=$((DFO + PFO))
TEMP=0.7
SEED=42
NUMSEQS=32
OUTLEN=512
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="$ROOT/experiments/paper_baselines/ssd_awq_70b_split_k1k2_exit_search"
OUT="$BASE/20260429_split_k${K}_K1_${K1}_K2_${K2}_dfo${DFO}_pfo${PFO}_exit${EXIT_LAYER}_temp07_seed${SEED}_ns${NUMSEQS}_all"

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
export SSD_FORCE_SPLIT_K1K2=1
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export SSD_DIST_PORT="${SSD_DIST_PORT:-12458}"

echo "[run_split_k1k2] exit=$EXIT_LAYER k=$K K1=$K1 K2=$K2 dfo=$DFO pfo=$PFO f=$F"
echo "[run_split_k1k2] numseqs=$NUMSEQS output_len=$OUTLEN temp=$TEMP seed=$SEED gpus=$CUDA_VISIBLE_DEVICES"
echo "[run_split_k1k2] out=$OUT"

(
  cd "$ROOT"
  SSD_PROFILE_DIR="$OUT" "$PY" -O bench/bench.py \
    --llama --size 8 \
    --model_path "$TARGET" \
    --draft_path "$DRAFT" \
    --quant_awq --quant_awq_artifact "$TGT_ART" --quant_group_size 128 \
    --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
    --async --spec \
    --k "$K" --f "$F" \
    --mesa --mesa_exit_layer "$EXIT_LAYER" \
    --mesa_phase1_k "$K1" --mesa_phase2_k "$K2" \
    --mesa_draft_fan_out "$DFO" --mesa_policy b \
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

"$PY" - "$OUT" "$BASE/proxy_wait_summary.csv" "$EXIT_LAYER" "$K" "$K1" "$K2" "$DFO" "$PFO" <<'PY'
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = Path(sys.argv[2])
exit_layer, k, k1, k2, dfo, pfo = map(int, sys.argv[3:])
run_name = out.name
log = (out / "run.log").read_text(errors="replace")

def grab(pattern: str):
    m = re.search(pattern, log)
    return m.group(1) if m else ""

profiles = sorted(out.glob("mesa_profile_draft_*.json"))
if not profiles:
    profiles = [Path(p) for p in glob.glob("/tmp/mesa_profile_draft_*.json")]
draft_profile = profiles[-1] if profiles else None
proxy_wait = []
phase1_replay = []
glue = []
phase2_replay = []
if draft_profile and draft_profile.exists():
    rows = json.load(open(draft_profile))
    proxy_wait = [float(r["ms"]) for r in rows if r.get("label") == "proxy_wait"]
    phase1_replay = [float(r["ms"]) for r in rows if r.get("label") == "phase1_replay"]
    phase2_replay = [float(r["ms"]) for r in rows if r.get("label") == "phase2_replay"]
    glue = [float(r["ms"]) for r in rows if r.get("label") == "glue"]

def med(xs):
    return statistics.median(xs) if xs else ""

row = {
    "run_name": run_name,
    "exit_layer": exit_layer,
    "k": k,
    "k1": k1,
    "k2": k2,
    "dfo": dfo,
    "pfo": pfo,
    "total_tps": grab(r"Total Throughput:\s*([\d.]+)"),
    "avg_tokens_per_step": grab(r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)"),
    "accept_fraction": grab(r"Avg Fraction.*Accepted:\s*([\d.]+)"),
    "p1_hit": grab(r"Avg Phase 1.*Hit Rate:\s*([\d.]+)"),
    "p2_hit": grab(r"Avg Phase 2.*Hit Rate:\s*([\d.]+)"),
    "p1_avg_accepted_len": grab(r"Avg Phase 1 Accepted Len:\s*([\d.]+)"),
    "p1_acceptance_ratio": grab(r"Avg Phase 1 Acceptance Ratio:\s*([\d.]+)"),
    "p2_avg_accepted_len": grab(r"Avg Phase 2 Accepted Len:\s*([\d.]+)"),
    "p2_acceptance_ratio": grab(r"Avg Phase 2 Acceptance Ratio:\s*([\d.]+)"),
    "target_verify_ms": grab(r"Avg target verify time.*:\s*([\d.]+)"),
    "draft_step_ms": grab(r"Avg draft step time.*:\s*([\d.]+)"),
    "proxy_wait_median_ms": med(proxy_wait),
    "proxy_wait_mean_ms": statistics.mean(proxy_wait) if proxy_wait else "",
    "proxy_wait_n": len(proxy_wait),
    "glue_median_ms": med(glue),
    "phase1_replay_median_ms": med(phase1_replay),
    "phase2_replay_median_ms": med(phase2_replay),
    "run_dir": str(out),
}
summary.parent.mkdir(parents=True, exist_ok=True)
old_rows = []
old_fields = []
if summary.exists():
    with summary.open(newline="") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
fieldnames = list(dict.fromkeys([*old_fields, *row.keys()]))
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(old_rows)
    writer.writerow(row)
print(f"-> appended {summary}")
print(f"proxy_wait_median_ms={row['proxy_wait_median_ms']}")
PY

(
  cd "$ROOT"
  "$PY" bench/plot_ssd_table.py "$BASE/proxy_wait_summary.csv" \
    --out "$BASE/proxy_wait_table.png" \
    --title "SSD AWQ 70B split K1/K2 exit search"
  "$PY" bench/plot_ssd_table.py "$BASE/proxy_wait_summary.csv" \
    --out "$BASE/proxy_perf_table.png" \
    --title "SSD AWQ 70B split K1/K2 proxy performance"
) >/dev/null 2>&1 || true

echo "[run_split_k1k2] done: $OUT"
