"""
Plot hazard metrics averaged across all output lengths (checkpoints).
4x1 layout: actual reject dist, recall@1, recall@3, recall@k
Both draft models in each subplot.
Style: consistent with corr_avg_selected.py (layerskip_overview annotation style)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 9, "figure.facecolor": "white",
    "mathtext.fontset": "cm",
})

DATA_PATH = "results/layerskip_70B_v4/correction_layerskip-llama2-70B.json"
OUT_PATH  = "results/layerskip_70B_v4/hazard_avg_across_cp.png"

with open(DATA_PATH) as f:
    data = json.load(f)

n_layers = data["n_layers"]  # 80
checkpoints = data["checkpoints"]
cp_labels = [f"cp{c}" for c in checkpoints]
drafts = data["drafts"]
layers = np.arange(n_layers + 1)  # 0..80

draft_names = ["TinyLlama-1.1B", "Llama2-7B"]
draft_colors = ["#2196F3", "#FF9800"]

W = 10  # window size


def avg_hazard_metric(draft_data, cp_labels, key):
    """Average a per-layer hazard metric across checkpoints, weighted by n_windows."""
    arrays, weights = [], []
    for cp in cp_labels:
        hz = draft_data[cp].get("hazard", {})
        nw = hz.get("n_windows", 0)
        if nw == 0:
            continue
        arrays.append(np.array(hz[key]))
        weights.append(nw)
    if not arrays:
        return None
    total_w = sum(weights)
    return sum(a * w for a, w in zip(arrays, weights)) / total_w


fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

# ── Panel 0: Actual reject position distribution ──
ax = axes[0]
bar_width = 0.35
for di, draft in enumerate(drafts):
    dd = draft["data"]
    reject_hist = np.zeros(W + 1)
    total = 0
    for cp in cp_labels:
        fd = dd[cp]
        for pos_str, cnt in fd.get("reject_pos_dist", {}).items():
            reject_hist[int(pos_str)] += cnt
        reject_hist[W] += fd.get("all_accept_count", 0)
        total += fd["count"] + fd.get("all_accept_count", 0)
    if total > 0:
        reject_hist /= total
    x = np.arange(W + 1) + di * bar_width
    ax.bar(x, reject_hist, bar_width, color=draft_colors[di], alpha=0.8,
           edgecolor="white", linewidth=0.5, label=draft_names[di])
ax.set_xticks(np.arange(W + 1) + bar_width / 2)
ax.set_xticklabels([str(i) for i in range(W)] + ["accept"], fontsize=9)
ax.set_ylabel("Fraction", fontsize=12)
ax.set_title("Reject Position Distribution", fontsize=13)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y")

# ── Panel 1: Recall@1 per layer ──
ax = axes[1]
for di, draft in enumerate(drafts):
    top1 = avg_hazard_metric(draft["data"], cp_labels, "top1_hit")
    if top1 is not None:
        ax.plot(layers, top1, color=draft_colors[di], linewidth=1.8,
                label=f"{draft_names[di]} (L50={top1[50]:.3f})")
ax.set_ylabel("Recall@1", fontsize=12)
ax.set_xlabel("Layer Index", fontsize=12)
ax.set_title(r"Recall@1:  $P(t^* \in \mathrm{top}_1(\hat{h}^{(l)}))$", fontsize=13)
ax.set_xlim(0, n_layers)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)

# ── Panel 2: Recall@3 per layer ──
ax = axes[2]
for di, draft in enumerate(drafts):
    top3 = avg_hazard_metric(draft["data"], cp_labels, "top3_hit")
    if top3 is not None:
        ax.plot(layers, top3, color=draft_colors[di], linewidth=1.8,
                label=f"{draft_names[di]} (L50={top3[50]:.3f})")
ax.set_ylabel("Recall@3", fontsize=12)
ax.set_xlabel("Layer Index", fontsize=12)
ax.set_title(r"Recall@3:  $P(t^* \in \mathrm{top}_3(\hat{h}^{(l)}))$", fontsize=13)
ax.set_xlim(0, n_layers)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)

# ── Panel 3: Recall@k at sample layers (both drafts) ──
ax = axes[3]
sample_layers_k = [40, 50, 60]
line_styles = ["-", "--", ":"]
for di, draft in enumerate(drafts):
    recall_at_k = avg_hazard_metric(draft["data"], cp_labels, "recall_at_k")  # [81][11]
    if recall_at_k is None:
        continue
    ks = np.arange(1, recall_at_k.shape[1] + 1)  # 1..11
    for si, sl in enumerate(sample_layers_k):
        ax.plot(ks, recall_at_k[sl], marker="o", markersize=4,
                color=draft_colors[di], linestyle=line_styles[si],
                linewidth=1.5, label=f"{draft_names[di]} L{sl}")
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel("Recall@k", fontsize=12)
ax.set_title(r"Recall@k:  $P(t^* \in \mathrm{top}_k(\hat{h}^{(l)}))$", fontsize=13)
ax.set_xlim(0.5, recall_at_k.shape[1] + 0.5)
ax.set_xticks(ks)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8.5, loc="lower right", ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    r"Hazard Reject Position Prediction"
    r"  ($\hat{h}_i^{(l)} = \prod_{j<i}\hat{\alpha}_j^{(l)}\,(1-\hat{\alpha}_i^{(l)})$,"
    r"  $\hat{\alpha}_i^{(l)} = \min(1,\, p_E^{(l)}(y_i)\,/\,p_D(y_i))$)"
    "\n"
    r"Target: LayerSkip-Llama2-70B  /  $D_1$: TinyLlama-1.1B,  $D_2$: Llama2-7B",
    fontsize=14, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
