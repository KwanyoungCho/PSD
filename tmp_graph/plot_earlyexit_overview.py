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

# X-axis: layer depth as percentage (0%=embedding, 100%=final)
x = np.linspace(0, 100, len(raw_jsd))

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

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Subplot 1: JSD ↓ ---
ax = axes[0]
ax.plot(x, raw_jsd, color="#1f77b4", lw=1.8, label="Llama-3.1-70B")
ax.axhline(draft_jsd_baseline, color="#ff7f0e", lw=1.5, ls="--",
           label=f"Llama-3.2-1B = {draft_jsd_baseline:.3f}")
ax.set_xlabel("Layer Depth (%)")
ax.set_ylabel("JSD ↓")
ax.set_title("JSD ↓")
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.75)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# --- Subplot 2: Top-5 Overlap ↑ ---
ax = axes[1]
ax.plot(x, topk_olap, color="#1f77b4", lw=1.8, label="Llama-3.1-70B")
# No reliable draft baseline for top-5 overlap
ax.set_xlabel("Layer Depth (%)")
ax.set_ylabel("Top-5 Overlap ↑")
ax.set_title("Top-5 Overlap ↑")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- Subplot 3: Top-5 Mass ↑ ---
ax = axes[2]
ax.plot(x, topk_mass, color="#1f77b4", lw=1.8, label="Llama-3.1-70B")
# No reliable draft baseline for top-5 mass
ax.set_xlabel("Layer Depth (%)")
ax.set_ylabel("Top-5 Mass ↑")
ax.set_title("Top-5 Mass ↑")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Llama-3.1-70B Early-Exit vs Final Distribution  (baseline: Llama-3.2-1B)",
    fontsize=14, y=1.02,
)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "earlyexit_overview.png")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
print(f"  Draft JSD baseline (weighted avg, {sum(draft_weights)} prompts): {draft_jsd_baseline:.4f}")
print(f"  NOTE: Top-5 Overlap / Mass baselines for Llama-3.2-1B not available in cp01 data.")
