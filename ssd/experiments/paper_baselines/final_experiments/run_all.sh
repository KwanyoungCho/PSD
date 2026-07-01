#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="$ROOT/experiments/paper_baselines/final_experiments"

PY="${PY:-/home/chokwans99/anaconda3/envs/ssd/bin/python}"
TARGET="${TARGET:-/data2/chokwans99/awq_calibrated/layerskip_llama2_70b}"
DRAFT_DENSE="${DRAFT_DENSE:-/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0}"
DRAFT_AWQ="${DRAFT_AWQ:-/data2/chokwans99/awq_calibrated/tinyllama_1b}"
TGT_ART="${TGT_ART:-/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4}"
DRAFT_ART="${DRAFT_ART:-/data2/chokwans99/awq_artifacts/tinyllama_1b/draft_tp1}"

TEMP="${TEMP:-0.7}"
SEED="${SEED:-42}"
NUMSEQS="${NUMSEQS:-50}"
INLEN="${INLEN:-512}"
OUTLEN="${OUTLEN:-512}"
PORT_BASE="${SSD_DIST_PORT:-12510}"

mkdir -p "$BASE"

export SSD_PROFILE_MESA="${SSD_PROFILE_MESA:-1}"
export SSD_CUDA_ARCH="${SSD_CUDA_ARCH:-8.6}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export SSD_HF_CACHE="${SSD_HF_CACHE:-/home/chokwans99/.cache/huggingface/hub}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/data2/chokwans99/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

postprocess() {
  local out="$1"
  local k="$2"
  (
    cd "$ROOT"
    "$PY" bench/summarize_ssd_run.py "$out" --k "$k" \
      --append-index "$BASE/summary_index.csv"
  ) >"$out/postprocess.log" 2>&1
}

copy_profiles() {
  local out="$1"
  local target_profile
  local draft_profile
  target_profile="$(grep -oE '/tmp/mesa_profile_target_rank0_[0-9]+\.json' "$out/run.log" | tail -1 || true)"
  draft_profile="$(grep -oE '/tmp/mesa_profile_draft_[0-9]+\.json' "$out/run.log" | tail -1 || true)"
  if [ -n "$target_profile" ] && [ -f "$target_profile" ]; then
    cp "$target_profile" "$out/"
  fi
  if [ -n "$draft_profile" ] && [ -f "$draft_profile" ]; then
    cp "$draft_profile" "$out/"
  fi
}

run_bench() {
  local label="$1"
  local gpus="$2"
  local port="$3"
  local k_for_summary="$4"
  shift 4

  local out="$BASE/20260430_${label}_temp07_seed${SEED}_ns${NUMSEQS}_in${INLEN}_out${OUTLEN}_all"
  mkdir -p "$out"
  if [ -s "$out/run.log" ]; then
    echo "run.log already exists: $out/run.log" >&2
    echo "move/remove it before rerunning." >&2
    exit 1
  fi

  echo "[final_experiments] START label=$label gpus=$gpus port=$port out=$out"
  echo "[final_experiments] temp=$TEMP seed=$SEED numseqs=$NUMSEQS input_len=$INLEN output_len=$OUTLEN dataset=all"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpus" SSD_DIST_PORT="$port" SSD_PROFILE_DIR="$out" \
      SSD_FORCE_SPLIT_K1K2="${SSD_FORCE_SPLIT_K1K2:-0}" \
      "$PY" -O bench/bench.py \
      --llama --size 8 \
      --model_path "$TARGET" \
      --quant_awq --quant_awq_artifact "$TGT_ART" --quant_group_size 128 \
      --gpus "${GPU_COUNT:-4}" --b 1 \
      --temp "$TEMP" --seed "$SEED" \
      --numseqs "$NUMSEQS" --input_len "$INLEN" --output_len "$OUTLEN" --all \
      --max_model_len 2048 \
      "$@"
  ) >"$out/run.log" 2>&1

  copy_profiles "$out"
  postprocess "$out" "$k_for_summary"
  echo "[final_experiments] DONE label=$label out=$out"
}

plot_final_table() {
  (
    cd "$ROOT"
    "$PY" - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/home/chokwans99/PSD/ssd/experiments/paper_baselines/final_experiments")
runs = [
    ("AR", "AR", "k=-", "target AWQ", "N/A", base / "20260430_ar_target_awq_temp07_seed42_ns50_in512_out512_all/summary_metrics.csv"),
    ("SD", "Sync SD", "k=7", "target AWQ", "draft AWQ", base / "20260430_sd_k7_target_awq_draft_awq_temp07_seed42_ns50_in512_out512_all/summary_metrics.csv"),
    ("SSD", "Async SSD", "k=7, f=3", "target AWQ", "draft AWQ", base / "20260430_ssd_k7_f3_target_awq_draft_awq_temp07_seed42_ns50_in512_out512_all/summary_metrics.csv"),
    ("Ours", "Split K1/K2", "K1=7, K2=5, dfo=2, pfo=1, exit=56", "target AWQ", "draft AWQ", base / "20260430_ours_k1_7_k2_5_dfo2_pfo1_exit56_temp07_seed42_ns50_in512_out512_all/summary_metrics.csv"),
]

rows = []
for label, mode_label, config, target, draft, path in runs:
    row = pd.read_csv(path).iloc[0].to_dict()
    rows.append({
        "Method": label,
        "Mode": mode_label,
        "Config": config,
        "Target": target,
        "Draft": draft,
        "Total tokens": row.get("total_tokens"),
        "TPS": row.get("total_tps"),
        "Decode TPS": row.get("decode_tps"),
        "Decode tokens": row.get("decode_total_tokens"),
        "Decode time s": row.get("decode_total_time_s"),
        "Avg tok/step": row.get("avg_tokens_per_step"),
        "Accept": row.get("accept_fraction"),
        "Cache hit": row.get("cache_hit_rate"),
        "P1 hit": row.get("p1_hit"),
        "P2 hit": row.get("p2_hit"),
        "P1 acc len": row.get("p1_avg_accepted_len"),
        "P2 acc len": row.get("p2_avg_accepted_len"),
        "Target verify ms": row.get("target_verify_ms"),
        "Target wait ms": row.get("prof_target_spec_wait_ms"),
        "Draft step ms": row.get("draft_step_ms"),
        "Proxy wait ms": row.get("prof_draft_proxy_wait_ms"),
        "Run dir": row.get("run_dir"),
    })

df = pd.DataFrame(rows)
ar_tps = float(df.loc[df["Method"] == "AR", "TPS"].iloc[0])
ar_decode = float(df.loc[df["Method"] == "AR", "Decode TPS"].iloc[0])
df["Speedup vs AR"] = df["TPS"].astype(float) / ar_tps
df["Decode speedup vs AR"] = df["Decode TPS"].astype(float) / ar_decode
df.to_csv(base / "summary_final.csv", index=False)
df.to_markdown(base / "summary_final.md", index=False)

headers = [
    "Method", "Config", "Target", "Draft", "TPS", "Speedup",
    "Decode TPS", "Decode speedup", "Avg tok/step", "Accept", "Cache hit",
    "P1 hit", "P2 hit", "Target verify", "Target wait", "Draft step", "Proxy wait",
]

def f2(x):
    if pd.isna(x):
        return "-"
    return f"{float(x):.2f}"

def ratio2(x):
    if pd.isna(x):
        return "-"
    return f"{float(x):.2f}"

table_rows = []
for _, row in df.iterrows():
    table_rows.append([
        row["Method"],
        row["Config"],
        row["Target"],
        row["Draft"],
        f2(row["TPS"]),
        f"{float(row['Speedup vs AR']):.2f}x",
        f2(row["Decode TPS"]),
        f"{float(row['Decode speedup vs AR']):.2f}x",
        f2(row["Avg tok/step"]),
        ratio2(row["Accept"]),
        ratio2(row["Cache hit"]),
        ratio2(row["P1 hit"]),
        ratio2(row["P2 hit"]),
        f2(row["Target verify ms"]),
        f2(row["Target wait ms"]),
        f2(row["Draft step ms"]),
        f2(row["Proxy wait ms"]),
    ])

fig, ax = plt.subplots(figsize=(23.5, 4.4))
ax.axis("off")
ax.set_title("Final Experiments: AR / SD / SSD / Ours (--all, numseqs=50, 512+512)", fontsize=18, fontweight="bold", pad=16)
table = ax.table(
    cellText=table_rows,
    colLabels=headers,
    cellLoc="center",
    colLoc="center",
    loc="center",
    bbox=[0, 0, 1, 0.84],
)
table.auto_set_font_size(False)
table.set_fontsize(8.2)
table.scale(1.0, 1.35)
widths = {
    0: 0.05, 1: 0.18, 2: 0.065, 3: 0.065, 4: 0.055, 5: 0.06,
    6: 0.065, 7: 0.075, 8: 0.065, 9: 0.05, 10: 0.055,
    11: 0.05, 12: 0.05, 13: 0.065, 14: 0.065, 15: 0.06, 16: 0.06,
}
for (r, c), cell in table.get_celld().items():
    if c in widths:
        cell.set_width(widths[c])
    cell.set_edgecolor("#29363a")
    cell.set_linewidth(0.45)
    if r == 0:
        cell.set_facecolor("#1f3a3d")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#f6f3ec" if r % 2 else "#ffffff")
        if table_rows[r - 1][0] == "Ours" and c in (4, 5, 6, 7):
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")

fig.savefig(base / "summary_final_table.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(base / "summary_final.csv")
print(base / "summary_final_table.png")
PY
  )
}

echo "[final_experiments] base=$BASE"

if [[ "${PLOT_ONLY:-0}" == "1" ]]; then
  plot_final_table
  exit 0
fi

if [[ "${SKIP_AR:-0}" != "1" ]]; then
  GPU_COUNT=4 run_bench "ar_target_awq" "0,1,2,3" "$PORT_BASE" 1 \
    --draft_path "$DRAFT_AWQ"
else
  echo "[final_experiments] SKIP_AR=1, reusing existing AR output"
fi

GPU_COUNT=4 run_bench "sd_k7_target_awq_draft_awq" "0,1,2,3" "$((PORT_BASE + 1))" 7 \
  --draft_path "$DRAFT_AWQ" \
  --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
  --gpu_memory_utilization 0.55 \
  --spec --k 7

GPU_COUNT=5 run_bench "ssd_k7_f3_target_awq_draft_awq" "0,1,2,3,4" "$((PORT_BASE + 2))" 7 \
  --draft_path "$DRAFT_AWQ" \
  --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
  --async --spec --k 7 --f 3

GPU_COUNT=5 SSD_FORCE_SPLIT_K1K2=1 run_bench "ours_k1_7_k2_5_dfo2_pfo1_exit56" "0,1,2,3,4" "$((PORT_BASE + 3))" 12 \
  --draft_path "$DRAFT_AWQ" \
  --quant_awq_draft --quant_awq_draft_artifact "$DRAFT_ART" \
  --async --spec --k 12 --f 3 \
  --mesa --mesa_exit_layer 56 \
  --mesa_phase1_k 7 --mesa_phase2_k 5 \
  --mesa_draft_fan_out 2 --mesa_policy b

plot_final_table

echo "[final_experiments] all done: $BASE"
