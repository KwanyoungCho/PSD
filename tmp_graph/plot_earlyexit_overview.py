#!/usr/bin/env python3
"""
Llama-3.1-70B Early-Exit vs Final Distribution Overview
Baseline: Llama-3.2-1B draft model
Style reference: layer_skip/layerskip_overview.png
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
RESULT = "/home/chokwans99/Parallel_SD/results/correction_Llama-3.1-70B-Instruct_cp01.json"
DEPRECATED = "/home/chokwans99/Parallel_SD/deprecated/Llama-3.1-70B-Instruct_results.json"
OUT_DIR = "/home/chokwans99/Parallel_SD/tmp_graph"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 9, "figure.facecolor": "white",
})

# ── Load cp01 data (cp1) ──
with open(RESULT) as f:
    data = json.load(f)

cp1 = data["data"]["cp1"]
n_layers = data["n_layers"]  # 80

raw_jsd = np.array(cp1["raw_jsd_layers"])       # 81 entries (layer 0..80)
topk_olap = np.array(cp1["topk_overlap_5"])      # 81 entries
topk_mass = np.array(cp1["topk_mass_5"])          # 81 entries

# X-axis: layer index
x = np.arange(len(raw_jsd))

# ── Draft baseline: JSD from deprecated data ──
# jsd_draft[-1] = JSD(draft_final, target_final), weighted avg over short/medium/long
with open(DEPRECATED) as f:
    dep = json.load(f)

draft_jsd_vals = []
draft_weights = []
for section in ["short", "medium", "long"]:
    sec = dep["results"][section]
    draft_jsd_vals.append(sec["jsd_draft"][-1])
    draft_weights.append(sec["num_prompts"])

draft_jsd_baseline = np.average(draft_jsd_vals, weights=draft_weights)

# ── Plot (JSD only, ~1:1 aspect) ──
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, raw_jsd, color="#1f77b4", lw=1.8, label="Llama-3.1-70B early-exit")
ax.axhline(draft_jsd_baseline, color="#ff7f0e", lw=1.5, ls="--",
           label="Llama-3.2-1B")
ax.set_xlabel("Layer Index")
ax.set_ylabel("JSD ↓")
ax.set_xlim(0, len(raw_jsd) - 1)
ax.set_ylim(0, 0.75)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title("Llama-3.1-70B vs. Llama-3.2-1B", fontsize=14)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "earlyexit_overview.png")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
print(f"  Draft JSD baseline (weighted avg, {sum(draft_weights)} prompts): {draft_jsd_baseline:.4f}")
print(f"  NOTE: Top-5 Overlap / Mass baselines for Llama-3.2-1B not available in cp01 data.")
