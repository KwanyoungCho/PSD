#!/usr/bin/env python3
"""
Revised LayerSkip vs Standard Early-Exit overview.
Changes from original:
  - X-axis: Layer Index (not Layer Depth %)
  - Title: "LayerSkip vs. Standard Early-Exit"
  - Mark crossover layer indices where LayerSkip surpasses each draft model
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
DATA_PATH = "/home/chokwans99/Parallel_SD/layer_skip/llama2_70B/layerskip_analysis.json"
OUT_DIR = "/home/chokwans99/Parallel_SD/tmp_graph"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 9, "figure.facecolor": "white",
})

# ── Load data ──
with open(DATA_PATH) as f:
    r = json.load(f)

t_ee = r["target_early_exit"]
ls_ee = r["layerskip_early_exit"]
drafts = r.get("draft_baselines") or []

t_name = os.path.basename(r["target"]).replace("/", "")
ls_name = os.path.basename(r["layerskip"]).replace("/", "")

draft_colors = ["#ff7f0e", "#2ca02c"]
draft_styles = ["--", ":"]


def find_crossover(vals, threshold, lower_is_better):
    """Find first index where vals crosses the threshold."""
    for i, v in enumerate(vals):
        if lower_is_better and v < threshold:
            return i
        if not lower_is_better and v > threshold:
            return i
    return None


# ── Metrics config ──
metrics_cfg = [
    ("jsd",          "JSD ↓",          (0, 0.75),  True),
    ("top5_overlap", "Top-5 Overlap ↑", (-0.02, 1.05), False),
    ("top5_mass",    "Top-5 Mass ↑",    (-0.02, 1.05), False),
]

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (key, ylabel, ylim, lower_better) in zip(axes, metrics_cfg):
    t_vals = np.array(t_ee[key])
    ls_vals = np.array(ls_ee[key])
    t_layers = np.arange(len(t_vals))
    ls_layers = np.arange(len(ls_vals))

    # Early-exit curves
    ax.plot(t_layers, t_vals, label=t_name, color="#1f77b4", linewidth=1.5)
    ax.plot(ls_layers, ls_vals, label=ls_name, color="#d62728", linewidth=1.5)

    # Draft baselines + crossover markers
    cross_indices = []
    for di, d in enumerate(drafts):
        bl = d[key]
        color = draft_colors[di % len(draft_colors)]
        ls_style = draft_styles[di % len(draft_styles)]
        ax.axhline(bl, color=color, linewidth=1.5, linestyle=ls_style,
                   label=f"{d['name']}")

        # Find crossover: first layer where LayerSkip surpasses this draft
        cross_idx = find_crossover(ls_vals, bl, lower_better)
        cross_indices.append(cross_idx)
        if cross_idx is not None:
            cross_val = ls_vals[cross_idx]
            # Marker at the crossover point
            ax.plot(cross_idx, cross_val, marker="v" if lower_better else "^",
                    markersize=9, color=color, zorder=5, markeredgecolor="white",
                    markeredgewidth=1.0)
            # Vertical dashed line
            ax.axvline(cross_idx, color=color, linewidth=0.9, linestyle=":",
                       alpha=0.5)
            # Annotation with offset to avoid overlap
            x_off = 5 if di == 0 else 5
            y_off = -16 if lower_better else 14
            if di == 1:
                y_off = y_off + (12 if not lower_better else -12)
            ax.annotate(f"Layer {cross_idx}",
                        xy=(cross_idx, cross_val),
                        xytext=(x_off, y_off),
                        textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=color, alpha=0.85, lw=0.8))

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, len(t_vals) - 1)
    ax.legend(fontsize=8.5, loc="upper right" if lower_better else "upper left")
    ax.grid(True, alpha=0.3)

fig.suptitle("LayerSkip vs. Standard Early-Exit", fontsize=14)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "layerskip_overview_revised.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# Print crossover summary
for key, ylabel, ylim, lower_better in metrics_cfg:
    ls_vals = ls_ee[key]
    print(f"\n{ylabel}:")
    for d in drafts:
        bl = d[key]
        idx = find_crossover(ls_vals, bl, lower_better)
        if idx is not None:
            print(f"  LayerSkip surpasses {d['name']} (={bl:.3f}) at layer {idx} (val={ls_vals[idx]:.4f})")
        else:
            print(f"  LayerSkip never surpasses {d['name']} (={bl:.3f})")
