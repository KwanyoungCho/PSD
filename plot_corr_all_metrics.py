"""
Comprehensive correction metrics plot — all indicators, both drafts, weighted avg across checkpoints.
Layout: 3 groups × rows, 2 cols (one per draft)
  Group 1 — Cover@k  : top1, top3, top5  (+ draft baseline hlines)
  Group 2 — Recall   : top3_recall, top5_recall  (+ draft baseline hlines)
  Group 3 — Mass     : mass_1, mass_3, mass_5  (+ draft baseline hlines)
  Group 4 — Distance : jsd, kl, tvd
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10})

DATA  = "/home/chokwans99/Parallel_SD/results/layerskip_70B_v4/correction_layerskip-llama2-70B.json"
OUT   = "/home/chokwans99/Parallel_SD/results/layerskip_70B_v4/corr_all_metrics.png"

with open(DATA) as f:
    data = json.load(f)

n_layers  = data["n_layers"]   # 80
cp_labels = [f"cp{c}" for c in data["checkpoints"]]
drafts    = data["drafts"]
layers    = np.arange(n_layers + 1)

DRAFT_NAMES  = ["TinyLlama-1.1B", "Llama2-7B"]
DRAFT_COLORS = ["#2196F3", "#FF9800"]


def wavg(dd, cp_labels, key, weight_key="count"):
    """Weighted average of a per-layer array across checkpoints."""
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
    """Weighted average of a scalar (possibly nested) across checkpoints."""
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


# ── Row specs: (metric_key, ylabel, ylim, draft_baseline_key or None) ──
rows = [
    # Cover
    ("corr_top1",        "Cover@1",         (-0.05, 1.05), ["draft_baseline", "top1"]),
    ("corr_top3",        "Cover@3",         (-0.05, 1.05), ["draft_baseline", "top3"]),
    ("corr_top5",        "Cover@5",         (-0.05, 1.05), ["draft_baseline", "top5"]),
    # Recall
    ("corr_top3_recall", "Recall@3",        (-0.05, 1.05), ["draft_baseline", "top3_recall"]),
    ("corr_top5_recall", "Recall@5",        (-0.05, 1.05), ["draft_baseline", "top5_recall"]),
    # Mass  — draft baseline for mass not in v4, skip
    ("corr_mass_1",      "Mass@1",          (-0.05, 1.05), None),
    ("corr_mass_3",      "Mass@3",          (-0.05, 1.05), None),
    ("corr_mass_5",      "Mass@5",          (-0.05, 1.05), None),
    # Distance
    ("corr_jsd",         "JSD(r̂, r_true) ↓", (0, 0.8),   ["draft_raw", "jsd"]),
    ("corr_tvd",         "TVD ↓",           (0, 1.0),      ["draft_raw", "tvd"]),
    ("corr_kl",          "KL(r_true ∥ r̂) ↓", None,       ["draft_raw", "kl"]),
]

n_rows = len(rows)
n_cols = 2  # one per draft

fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows), squeeze=False)

for col, (draft, dname, dcolor) in enumerate(zip(drafts, DRAFT_NAMES, DRAFT_COLORS)):
    dd = draft["data"]
    total_c = sum(dd[cp]["count"] for cp in cp_labels)

    for row, (key, ylabel, ylim, bl_path) in enumerate(rows):
        ax = axes[row, col]

        # Per-layer weighted average
        vals = wavg(dd, cp_labels, key)
        if vals is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.plot(layers, vals, color=dcolor, linewidth=1.8, label="Early-exit residual")

        # Vertical reference lines at L50, L60, L70
        for ref_l, ref_ls in [(50, "--"), (60, ":"), (70, "-.")]:
            ax.axvline(ref_l, color="gray", lw=0.7, ls=ref_ls, alpha=0.5)
            ax.text(ref_l + 0.5, ax.get_ylim()[0] if ylim is None else ylim[0],
                    f"L{ref_l}", fontsize=7, color="gray", va="bottom")

        # Draft baseline hline
        if bl_path is not None:
            bl = scalar_wavg(dd, cp_labels, bl_path)
            if bl is not None:
                ax.axhline(bl, color="#e63946", lw=1.4, ls="--", alpha=0.85,
                           label=f"Draft baseline = {bl:.3f}")

        # Annotate L50 value
        ax.annotate(f"L50: {vals[50]:.3f}", xy=(50, vals[50]),
                    xytext=(52, vals[50] + (0.03 if ylim and ylim[1] > 0.5 else -0.03)),
                    fontsize=7.5, color=dcolor,
                    arrowprops=dict(arrowstyle="-", color=dcolor, lw=0.8))

        if row == 0:
            ax.set_title(f"{dname}  (n_reject_avg={total_c})", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlabel("Layer index $l$" if row == n_rows - 1 else "", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(-1, n_layers + 1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

# Group separators
group_starts = [0, 3, 5, 8]
group_labels = ["Cover@k  (argmax(r_true) ∈ topk(r̂))",
                "Recall@k  (set overlap topk(r_true) ∩ topk(r̂))",
                "Mass@k  (Σ r_true[topk(r̂)])",
                "Distribution Distance  (r̂ vs r_true)"]
for gs, gl in zip(group_starts, group_labels):
    for col in range(n_cols):
        axes[gs, col].set_facecolor("#f5f5f5")

# Add group annotations on the left
for gs, gl in zip(group_starts, group_labels):
    axes[gs, 0].annotate(
        gl, xy=(-0.28, 0.5), xycoords="axes fraction",
        fontsize=9, fontweight="bold", color="#333",
        rotation=90, ha="center", va="center",
    )

fig.suptitle(
    "Correction Distribution Metrics — All Indicators\n"
    "Target: LayerSkip-Llama2-70B  |  Weighted avg across output lengths (cp1/64/128/192/256)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0.03, 0, 1, 0.97])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
