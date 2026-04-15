"""
Correction metrics averaged across checkpoints (weighted by count).
Metrics: Top-1 Match, Top-3 Cover, Top-5 Recall
Layout: 1×3 (horizontal)
Baseline: draft baseline (horizontal line)
Annotation style: marker + vertical dashed line + boxed label (layerskip_overview style)
Legend style: corr_combined style with math labels
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

DATA = "results/layerskip_70B_v4/correction_layerskip-llama2-70B.json"
OUT  = "results/layerskip_70B_v4/corr_avg_selected.png"

with open(DATA) as f:
    data = json.load(f)

n_layers  = data["n_layers"]  # 80
cp_labels = [f"cp{c}" for c in data["checkpoints"]]
drafts    = data["drafts"]
layers    = np.arange(n_layers + 1)

DRAFT_NAMES  = ["TinyLlama-1.1B", "Llama2-7B"]
DRAFT_COLORS = ["#2196F3", "#FF9800"]

METRICS = [
    # (corr_key, baseline_key, ylabel, title)
    ("corr_top1",        ["draft_baseline", "top1"],        "Top-1 Match ↑",
     r"Top-1 Match:  $P(\arg\max\, r \in \mathrm{top}_1(\hat{r}))$"),
    ("corr_top3",        ["draft_baseline", "top3"],        "Top-3 Cover ↑",
     r"Top-3 Cover:  $P(\arg\max\, r \in \mathrm{top}_3(\hat{r}))$"),
    ("corr_top5_recall", ["draft_baseline", "top5_recall"], "Top-5 Recall ↑",
     r"Top-5 Recall:  $|\mathrm{top}_5(r) \cap \mathrm{top}_5(\hat{r})|\,/\,5$"),
]


def wavg(dd, cp_labels, key, weight_key="count"):
    arr, wts = [], []
    for cp in cp_labels:
        fd = dd[cp]
        w = fd[weight_key]
        if w == 0:
            continue
        arr.append(np.array(fd[key], dtype=float))
        wts.append(w)
    if not arr:
        return None
    total = sum(wts)
    return sum(a * w for a, w in zip(arr, wts)) / total


def scalar_wavg(dd, cp_labels, key_path, weight_key="count"):
    vals, wts = [], []
    for cp in cp_labels:
        fd = dd[cp]
        w = fd[weight_key]
        if w == 0:
            continue
        v = fd
        for k in key_path:
            v = v[k]
        vals.append(float(v))
        wts.append(w)
    if not vals:
        return None
    total = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / total


def find_first_crossover(vals, baseline):
    """Find the first layer where vals >= baseline."""
    for i, v in enumerate(vals):
        if v >= baseline:
            return i
    return None


fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5))

for col, (corr_key, bl_path, ylabel, title) in enumerate(METRICS):
    ax = axes[col]

    for di, (draft, dname, dcolor) in enumerate(zip(drafts, DRAFT_NAMES, DRAFT_COLORS)):
        dd = draft["data"]
        sub = str(di + 1)

        vals = wavg(dd, cp_labels, corr_key)
        bl = scalar_wavg(dd, cp_labels, bl_path)
        if vals is None or bl is None:
            continue

        # Correction curve (solid)
        ax.plot(layers, vals, color=dcolor, linewidth=1.8,
                label=f"Early-exit residual ({dname})")

        # Draft baseline (dashed)
        ax.axhline(bl, color=dcolor, linestyle="--", linewidth=1.3, alpha=0.7,
                    label=f"Draft: {dname} ({bl:.3f})")

        # Find first crossover layer
        cross_l = find_first_crossover(vals, bl)
        if cross_l is not None:
            cross_val = vals[cross_l]

            # Marker at crossover
            ax.plot(cross_l, cross_val, marker="^", markersize=9,
                    color=dcolor, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.0)

            # Vertical dashed line
            ax.axvline(cross_l, color=dcolor, linewidth=0.9, linestyle=":",
                       alpha=0.5)

            # Boxed annotation with offset to avoid overlap
            y_off = 14 if di == 0 else 26
            ax.annotate(
                f"Layer {cross_l}",
                xy=(cross_l, cross_val),
                xytext=(5, y_off),
                textcoords="offset points",
                fontsize=8.5, fontweight="bold", color=dcolor,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=dcolor, alpha=0.85, lw=0.8),
            )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_xlim(0, n_layers)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

fig.suptitle(
    r"Correction Metrics  ($r = p_{\mathrm{target}} - p_D$,  $\hat{r}^{(l)} = p_E^{(l)} - p_D$)"
    "\n"
    r"Target: LayerSkip-Llama2-70B  /  $D_1$: TinyLlama-1.1B,  $D_2$: Llama2-7B",
    fontsize=14, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
