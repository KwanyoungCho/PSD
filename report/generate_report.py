#!/usr/bin/env python3
"""Generate comprehensive correction analysis report with all plots."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULT_DIR = "/home/chokwans99/Parallel_SD/results"
REPORT_DIR = "/home/chokwans99/Parallel_SD/report"
os.makedirs(REPORT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

MODELS = {
    "Llama-3.1-70B-Instruct": {
        "file": "correction_Llama-3.1-70B-Instruct.json",
        "short": "Llama 70B+1B",
        "color_map": {"cp0": "#1f77b4", "cp128": "#ff7f0e", "cp256": "#2ca02c", "cp512": "#d62728"},
    },
    "Qwen3-32B": {
        "file": "correction_Qwen3-32B.json",
        "short": "Qwen3 32B+0.6B",
        "color_map": {"cp0": "#1f77b4", "cp128": "#ff7f0e", "cp256": "#2ca02c", "cp512": "#d62728"},
    },
}

# ── Load data ──
all_data = {}
for model_name, meta in MODELS.items():
    with open(os.path.join(RESULT_DIR, meta["file"])) as f:
        all_data[model_name] = json.load(f)


# ============================================================
# Fig 1: JSD across all layers (per checkpoint, per model)
# ============================================================
def plot_jsd_layers(model_name, data, meta):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_layers = data["n_layers"]

    for cp in data["data"]:
        jsd = data["data"][cp]["jsd_layers"]
        layers = np.arange(len(jsd))  # 0 = embedding, 1..N = transformer layers
        color = meta["color_map"][cp]
        cnt = data["data"][cp]["count"]
        ax.plot(layers, jsd, label=f"{cp} (n_reject={cnt})", color=color, linewidth=1.5)

    ax.set_xlabel("Target Layer Index (0=embedding)")
    ax.set_ylabel("JSD(r_true, r̂_k)")
    ax.set_title(f"{meta['short']} — JSD of Correction Distribution vs Early-Exit Proxy (all layers)")
    ax.legend()
    ax.set_xlim(0, n_layers)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig1_jsd_layers_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 2: JSD zoomed into last 20 layers
# ============================================================
def plot_jsd_last_layers(model_name, data, meta, last_n=20):
    fig, ax = plt.subplots(figsize=(10, 5))
    n_layers = data["n_layers"]
    n_total = n_layers + 1  # includes embedding layer
    start = max(0, n_total - last_n)

    for cp in data["data"]:
        jsd = data["data"][cp]["jsd_layers"][start:]
        layers = np.arange(start, start + len(jsd))
        color = meta["color_map"][cp]
        cnt = data["data"][cp]["count"]
        ax.plot(layers, jsd, marker="o", markersize=4, label=f"{cp} (n={cnt})", color=color, linewidth=1.5)

    ax.set_xlabel("Target Layer Index")
    ax.set_ylabel("JSD(r_true, r̂_k)")
    ax.set_title(f"{meta['short']} — JSD (last {last_n} layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig2_jsd_zoom_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 3: Top-1 match across layers
# ============================================================
def plot_top1_layers(model_name, data, meta):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_layers = data["n_layers"]

    for cp in data["data"]:
        vals = [v * 100 for v in data["data"][cp]["top1_match"]]
        layers = np.arange(len(vals))
        color = meta["color_map"][cp]
        cnt = data["data"][cp]["count"]
        ax.plot(layers, vals, label=f"{cp} (n={cnt})", color=color, linewidth=1.5)

    ax.set_xlabel("Target Layer Index (0=embedding)")
    ax.set_ylabel("Top-1 Match (%)")
    ax.set_title(f"{meta['short']} — Top-1 Match: Proxy vs True Correction Token")
    ax.legend()
    ax.set_xlim(0, n_layers)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig3_top1_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 4: Top-5 coverage across layers
# ============================================================
def plot_top5_cover_layers(model_name, data, meta):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_layers = data["n_layers"]

    for cp in data["data"]:
        vals = [v * 100 for v in data["data"][cp]["top5_cover"]]
        layers = np.arange(len(vals))
        color = meta["color_map"][cp]
        cnt = data["data"][cp]["count"]
        ax.plot(layers, vals, label=f"{cp} (n={cnt})", color=color, linewidth=1.5)

    ax.set_xlabel("Target Layer Index (0=embedding)")
    ax.set_ylabel("Top-5 Coverage (%)")
    ax.set_title(f"{meta['short']} — Top-5 Coverage: True Correction Token in Proxy Top-5")
    ax.legend()
    ax.set_xlim(0, n_layers)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig4_top5cover_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 5: Top-5 recall across layers
# ============================================================
def plot_top5_recall_layers(model_name, data, meta):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_layers = data["n_layers"]

    for cp in data["data"]:
        vals = [v * 100 for v in data["data"][cp]["top5_recall"]]
        layers = np.arange(len(vals))
        color = meta["color_map"][cp]
        cnt = data["data"][cp]["count"]
        ax.plot(layers, vals, label=f"{cp} (n={cnt})", color=color, linewidth=1.5)

    ax.set_xlabel("Target Layer Index (0=embedding)")
    ax.set_ylabel("Top-5 Recall (%)")
    ax.set_title(f"{meta['short']} — Top-5 Recall: Overlap of Proxy Top-5 with True Residual Top-5")
    ax.legend()
    ax.set_xlim(0, n_layers)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig5_top5recall_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 6: Reject position distribution (bar chart)
# ============================================================
def plot_reject_pos_dist(model_name, data, meta):
    cps = list(data["data"].keys())
    n_cp = len(cps)
    fig, axes = plt.subplots(1, n_cp, figsize=(4 * n_cp, 4), sharey=True)
    if n_cp == 1:
        axes = [axes]

    positions = list(range(10))
    for idx, cp in enumerate(cps):
        ax = axes[idx]
        dist = data["data"][cp]["reject_pos_dist"]
        counts = [dist.get(str(p), 0) for p in positions]
        total = sum(counts)
        pcts = [c / total * 100 if total > 0 else 0 for c in counts]
        ax.bar(positions, pcts, color=meta["color_map"][cp], alpha=0.8)
        ax.set_title(f"{cp} (n={total})")
        ax.set_xlabel("Draft Token Position")
        if idx == 0:
            ax.set_ylabel("First Reject (%)")
        ax.set_xticks(positions)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{meta['short']} — First Rejection Position Distribution", fontsize=14)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig6_reject_pos_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 7: Summary comparison bar chart (proxy L-2 vs draft baseline)
# ============================================================
def plot_proxy_vs_draft(model_name, data, meta):
    cps = list(data["data"].keys())
    metrics = ["top1", "top5_cover", "top5_recall"]
    labels = ["Top-1 Match", "Top-5 Coverage", "Top-5 Recall"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(cps))
    width = 0.35

    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[mi]
        proxy_vals = []
        draft_vals = []
        for cp in cps:
            info = data["data"][cp]
            # proxy L[-2] (second to last)
            if metric == "top1":
                proxy_vals.append(info["top1_match"][-2] * 100)
                draft_vals.append(info["draft_baseline"]["top1_match"] * 100)
            elif metric == "top5_cover":
                proxy_vals.append(info["top5_cover"][-2] * 100)
                draft_vals.append(info["draft_baseline"]["top5_cover"] * 100)
            elif metric == "top5_recall":
                proxy_vals.append(info["top5_recall"][-2] * 100)
                draft_vals.append(info["draft_baseline"]["top5_recall"] * 100)

        bars1 = ax.bar(x - width / 2, proxy_vals, width, label="Proxy L[-2]", color="#2196F3", alpha=0.8)
        bars2 = ax.bar(x + width / 2, draft_vals, width, label="Draft Baseline", color="#FF9800", alpha=0.8)

        # value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(cps)
        ax.set_ylabel("%")
        ax.set_title(label)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{meta['short']} — Early-Exit Proxy L[-2] vs Draft Baseline", fontsize=14)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig7_proxy_vs_draft_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 8: Accept probability & reject rate across checkpoints
# ============================================================
def plot_accept_reject_overview(model_name, data, meta):
    cps = list(data["data"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    reject_rates = []
    accept_probs = []
    first_reject_pos = []
    for cp in cps:
        info = data["data"][cp]
        cnt = info["count"]
        aa = info["all_accept_count"]
        total = cnt + aa
        reject_rates.append(cnt / total * 100)
        accept_probs.append(info["mean_accept_prob"] * 100)
        first_reject_pos.append(info["mean_first_reject"])

    x = np.arange(len(cps))
    colors = [meta["color_map"][cp] for cp in cps]

    ax = axes[0]
    ax.bar(x, reject_rates, color=colors, alpha=0.8)
    for i, v in enumerate(reject_rates):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cps)
    ax.set_ylabel("Reject Rate (%)")
    ax.set_title("Reject Rate")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, accept_probs, color=colors, alpha=0.8)
    for i, v in enumerate(accept_probs):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cps)
    ax.set_ylabel("Mean Accept Prob (%)")
    ax.set_title("Mean Accept Probability")
    ax.set_ylim(0, max(accept_probs) * 1.3 + 5)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(x, first_reject_pos, color=colors, alpha=0.8)
    for i, v in enumerate(first_reject_pos):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cps)
    ax.set_ylabel("Position")
    ax.set_title("Mean First Reject Position")
    ax.set_ylim(0, max(first_reject_pos) * 1.3 + 0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{meta['short']} — SD Acceptance Overview", fontsize=14)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig8_overview_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 9: Cross-model comparison (side by side for each cp)
# ============================================================
def plot_cross_model_jsd():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cps = ["cp0", "cp128", "cp256", "cp512"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for mi, (model_name, meta) in enumerate(MODELS.items()):
        ax = axes[mi]
        data = all_data[model_name]
        n_layers = data["n_layers"]

        for ci, cp in enumerate(cps):
            if cp not in data["data"]:
                continue
            jsd = data["data"][cp]["jsd_layers"]
            layers = np.arange(len(jsd))
            # normalize to [0,100] x-axis for comparison
            ax.plot(layers / n_layers * 100, jsd, label=cp, color=colors[ci], linewidth=1.5)

        ax.set_xlabel("Layer Depth (%)")
        ax.set_ylabel("JSD")
        ax.set_title(f"{meta['short']} ({n_layers} layers)")
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Model JSD Comparison (normalized layer depth)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "fig9_cross_model_jsd.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Fig 10: Heatmap — top5 coverage across layers and checkpoints
# ============================================================
def plot_top5_heatmap(model_name, data, meta):
    cps = list(data["data"].keys())
    n_layers = data["n_layers"]

    matrix = []
    for cp in cps:
        vals = [v * 100 for v in data["data"][cp]["top5_cover"]]
        matrix.append(vals)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(cps)))
    ax.set_yticklabels(cps)
    ax.set_xlabel("Layer Index")
    ax.set_title(f"{meta['short']} — Top-5 Coverage Heatmap (% across layers)")

    # show every 10th layer tick
    n_total = len(matrix[0])  # n_layers + 1 (includes embedding)
    tick_step = 10 if n_total > 40 else 5
    ax.set_xticks(range(0, n_total, tick_step))
    ax.set_xticklabels(range(0, n_total, tick_step))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Top-5 Coverage (%)")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"fig10_heatmap_{model_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# Generate all plots
# ============================================================
all_figures = []
for model_name, meta in MODELS.items():
    data = all_data[model_name]
    all_figures.append(plot_jsd_layers(model_name, data, meta))
    all_figures.append(plot_jsd_last_layers(model_name, data, meta))
    all_figures.append(plot_top1_layers(model_name, data, meta))
    all_figures.append(plot_top5_cover_layers(model_name, data, meta))
    all_figures.append(plot_top5_recall_layers(model_name, data, meta))
    all_figures.append(plot_reject_pos_dist(model_name, data, meta))
    all_figures.append(plot_proxy_vs_draft(model_name, data, meta))
    all_figures.append(plot_accept_reject_overview(model_name, data, meta))
    all_figures.append(plot_top5_heatmap(model_name, data, meta))

all_figures.append(plot_cross_model_jsd())

print(f"Generated {len(all_figures)} figures:")
for f in all_figures:
    print(f"  {os.path.basename(f)}")


# ============================================================
# Generate Markdown report
# ============================================================
def build_report():
    lines = []
    lines.append("# MESA-SSD Correction Distribution Analysis Report")
    lines.append("")
    lines.append("**Experiment**: Can the correction distribution at a rejected SD position")
    lines.append("be approximated by an early-exit (logit lens) proxy from an intermediate target layer?")
    lines.append("")
    lines.append("- **Dataset**: GSM8K (200 samples)")
    lines.append("- **Window size**: 10 draft tokens")
    lines.append("- **Checkpoints**: cp0, cp128, cp256, cp512 (number of target-generated context tokens)")
    lines.append("- **Rejection**: Probabilistic SD — accept with prob min(1, p_T(y)/p_D(y))")
    lines.append("- **Metrics computed at first reject position only**")
    lines.append("")
    lines.append("---")
    lines.append("")

    for model_name, meta in MODELS.items():
        data = all_data[model_name]
        n_layers = data["n_layers"]
        short = meta["short"]

        lines.append(f"## {short}")
        lines.append("")
        lines.append(f"- Target layers: {n_layers}")
        lines.append(f"- Samples: {data['n_samples']}")
        lines.append("")

        # Summary table
        lines.append("### Overview")
        lines.append("")
        lines.append(f"![Accept/Reject Overview](fig8_overview_{model_name}.png)")
        lines.append("")
        lines.append("| Checkpoint | Valid Samples | Rejected | All-Accepted | Reject Rate | Accept Prob | First Reject Pos |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for cp in data["data"]:
            info = data["data"][cp]
            cnt = info["count"]
            aa = info["all_accept_count"]
            total = cnt + aa
            rr = cnt / total * 100
            ap = info["mean_accept_prob"]
            frp = info["mean_first_reject"]
            lines.append(f"| {cp} | {total} | {cnt} | {aa} | {rr:.1f}% | {ap:.4f} | {frp:.2f} |")
        lines.append("")

        # Reject position distribution
        lines.append("### Rejection Position Distribution")
        lines.append("")
        lines.append(f"![Reject Position](fig6_reject_pos_{model_name}.png)")
        lines.append("")
        lines.append("Shows where in the 10-token draft window the first rejection occurs.")
        lines.append("Position 0 = first draft token.")
        lines.append("")

        # JSD
        lines.append("### JSD: Correction Distribution vs Early-Exit Proxy")
        lines.append("")
        lines.append(f"![JSD All Layers](fig1_jsd_layers_{model_name}.png)")
        lines.append("")
        lines.append(f"![JSD Last 20 Layers](fig2_jsd_zoom_{model_name}.png)")
        lines.append("")

        # Key JSD values table
        lines.append("**JSD at key layers:**")
        lines.append("")
        header = "| Checkpoint |"
        for li in [-5, -4, -3, -2, -1]:
            layer_idx = n_layers + li + 1
            header += f" L{layer_idx} |"
        lines.append(header)
        lines.append("|:---:|" + ":---:|" * 5)
        for cp in data["data"]:
            jsd = data["data"][cp]["jsd_layers"]
            row = f"| {cp} |"
            for li in [-5, -4, -3, -2, -1]:
                row += f" {jsd[li]:.4f} |"
            lines.append(row)
        lines.append("")

        # Top-1 match
        lines.append("### Top-1 Match")
        lines.append("")
        lines.append("Whether the proxy's argmax correction token matches the true correction token.")
        lines.append("")
        lines.append(f"![Top-1 Match](fig3_top1_{model_name}.png)")
        lines.append("")

        # Top-5 coverage
        lines.append("### Top-5 Coverage")
        lines.append("")
        lines.append("Whether the true correction token falls within the proxy's top-5.")
        lines.append("")
        lines.append(f"![Top-5 Coverage](fig4_top5cover_{model_name}.png)")
        lines.append("")
        lines.append(f"![Top-5 Heatmap](fig10_heatmap_{model_name}.png)")
        lines.append("")

        # Top-5 recall
        lines.append("### Top-5 Recall")
        lines.append("")
        lines.append("Overlap ratio: |proxy_top5 ∩ true_top5| / |true_top5|")
        lines.append("")
        lines.append(f"![Top-5 Recall](fig5_top5recall_{model_name}.png)")
        lines.append("")

        # Proxy vs Draft baseline comparison
        lines.append("### Early-Exit Proxy L[-2] vs Draft Baseline")
        lines.append("")
        lines.append(f"![Proxy vs Draft](fig7_proxy_vs_draft_{model_name}.png)")
        lines.append("")
        lines.append("| Checkpoint | Proxy L[-2] Top1 | Draft Top1 | Proxy L[-2] Top5 | Draft Top5 | Proxy L[-2] Top5R | Draft Top5R |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for cp in data["data"]:
            info = data["data"][cp]
            db = info["draft_baseline"]
            pt1 = info["top1_match"][-2] * 100
            dt1 = db["top1_match"] * 100
            pt5 = info["top5_cover"][-2] * 100
            dt5 = db["top5_cover"] * 100
            pr5 = info["top5_recall"][-2] * 100
            dr5 = db["top5_recall"] * 100
            lines.append(f"| {cp} | {pt1:.1f}% | {dt1:.1f}% | {pt5:.1f}% | {dt5:.1f}% | {pr5:.1f}% | {dr5:.1f}% |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Cross-model comparison
    lines.append("## Cross-Model Comparison")
    lines.append("")
    lines.append("![Cross-Model JSD](fig9_cross_model_jsd.png)")
    lines.append("")

    # Key findings
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("### Llama 70B + 1B")
    lines.append("")
    lines.append("1. **Early-exit proxy is highly effective at cp0**: L79 (second-to-last) achieves")
    lines.append("   94.5% top-5 coverage and 73.9% top-1 match for the correction token.")
    lines.append("   JSD drops to 0.136 — the proxy closely approximates the true correction distribution.")
    lines.append("")
    lines.append("2. **Proxy outperforms draft baseline at cp0**: Draft top-5 coverage is only 32.7%")
    lines.append("   vs proxy's 94.5% — a ~3x improvement, validating the MESA-SSD hypothesis.")
    lines.append("")
    lines.append("3. **Longer context reduces proxy advantage**: At cp128+, reject rate drops (77.6%→45.2%)")
    lines.append("   and remaining rejections are harder cases where even the proxy struggles.")
    lines.append("   At cp512, draft baseline (78.6%) actually surpasses the proxy (14.3%).")
    lines.append("")
    lines.append("4. **Rejection concentrates at position 0**: At cp0, 79% of rejections occur at")
    lines.append("   the first draft token. With more context, rejections spread across the window.")
    lines.append("")
    lines.append("5. **Statistical caveat**: cp256 has only 20 rejections, cp512 has 14 — interpret with care.")
    lines.append("")
    lines.append("### Qwen3 32B + 0.6B")
    lines.append("")
    lines.append("1. **Fundamental draft-target mismatch**: Accept probability ≈ 0 across all checkpoints.")
    lines.append("   The 0.6B draft model's distribution barely overlaps with the 32B target.")
    lines.append("")
    lines.append("2. **Early-exit proxy fails completely**: JSD remains ~0.64–0.69 even at L63")
    lines.append("   (second-to-last). Top-5 coverage is 0.5–9.6%. The proxy provides almost")
    lines.append("   no useful information about the correction distribution.")
    lines.append("")
    lines.append("3. **Draft baseline also fails**: Both proxy and draft baselines are near 0%,")
    lines.append("   indicating the correction distribution is unpredictable from either source.")
    lines.append("")
    lines.append("4. **Implication**: MESA-SSD requires draft and target models from the same family")
    lines.append("   with sufficient representation similarity. Qwen3-0.6B/32B pairing is too distant.")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append("- The MESA-SSD hypothesis (early-exit layers can predict correction tokens) is")
    lines.append("  **confirmed for Llama** and **rejected for Qwen3** with current model pairings.")
    lines.append("- The proxy is most valuable in low-context scenarios where draft-target divergence is high.")
    lines.append("- A cp1 experiment (1 target token of context) is needed to validate under realistic")
    lines.append("  SD conditions, since cp0 is artificially harsh (no bonus token from prefill).")
    lines.append("")

    return "\n".join(lines)


report_md = build_report()
report_path = os.path.join(REPORT_DIR, "report.md")
with open(report_path, "w") as f:
    f.write(report_md)
print(f"\nReport written to: {report_path}")
