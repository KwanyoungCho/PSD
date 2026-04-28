"""3-way breakdown plots with DISTINCT non-overlapping colors.

Same data sources as plot_3way.py but uses tab20 + tab20b palette for
unique colors per phase, plus hatching to disambiguate when colors are
close. Phases are sorted to put related families adjacent.
"""
import os
import sys
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/home/chokwans99/PSD/ssd/experiments/hybrid_vs_split_70b/results"
# Default baseline = pre-hybrid BOTH-AWQ run (target + draft both quantized),
# matching the new experiment default. Override with argv[2] when comparing
# against an older dense-draft baseline.
BASELINE = sys.argv[2] if len(sys.argv) > 2 else "/home/chokwans99/PSD/ssd/experiments/quant_awq/70b/draft_awq/mesa_k5_f4_dfo2_exit40"

CONFIGS = [
    ("(A) pre-hybrid\nsplit baseline", BASELINE),
    ("(B) current code\nsplit fallback", f"{RESULTS}/split_k5_dfo2_exit40"),
    ("(C) current code\nhybrid (K1=3,K2=2)", f"{RESULTS}/hybrid_k5_K1_3_K2_2_exit40"),
]

# Hand-curated distinct colors. Group by family, vary brightness within family.
# Using tab10 / Set1 / Set2 mixed; verified against colorblind palette tools.
PHASE_STYLE = {
    # ---- DRAFT side ----
    # phase1 family (blue tones)
    "phase1_replay":          ("#08306b", ""),    # navy
    "phase1_build":           ("#4292c6", ""),    # blue
    "phase1_prep":            ("#9ecae1", "//"),  # light blue + hatch
    # phase2 split family (green tones)
    "phase2_replay":          ("#00441b", ""),    # dark green
    "phase2_build":           ("#41ab5d", ""),    # green
    "phase2_prep":            ("#a1d99b", "//"),  # light green + hatch
    # phase2 hybrid family (orange tones)
    "phase2_hybrid_replay_long":  ("#a63603", ""),
    "phase2_hybrid_replay_short": ("#fd8d3c", "xx"),
    "phase2_hybrid_build":        ("#fdae6b", "\\\\"),
    "phase2_hybrid_eager_long":   ("#7f2704", "++"),
    "phase2_hybrid_eager_short":  ("#fdd0a2", "OO"),
    # glue/cache (cyan/teal)
    "glue":                   ("#01665e", ""),
    "draft_glue_replay":      ("#5ab4ac", ""),
    "hit_cache_respond":      ("#80cdc1", "//"),
    # IO/wait (gray tones)
    "draft_recv_cmd":         ("#525252", ""),
    "draft_send_response":    ("#969696", ""),
    "proxy_wait":             ("#bdbdbd", "..."),
    "merge_cache":            ("#000000", ""),
    "tree_prep":              ("#737373", "xx"),
    "tree_replay":            ("#252525", ""),
    # ---- TARGET side ----
    # graph (red tones)
    "graph_pre":              ("#67000d", ""),
    "graph_post":             ("#cb181c", ""),
    # wait
    "target_spec_wait":       ("#fdd49e", ""),
    "target_postprocess":     ("#fee391", ".."),
    # logits/setup (purple tones)
    "exit_logits":            ("#3f007d", ""),
    "final_logits":           ("#807dba", "//"),
    "verify_setup":           ("#dadaeb", "xx"),
    # verify
    "verify_replay":          ("#990000", ""),
    "verify_sample_accept":   ("#ec7014", ""),
    "proxy_compute_send":     ("#fec44f", ""),
}

# Fallback for unseen labels
FALLBACK = ("#bdc3c7", "")


def _style(label):
    return PHASE_STYLE.get(label, FALLBACK)


def _load(cfg_path):
    p = f"{cfg_path}/mesa_per_step_contribution.csv"
    return pd.read_csv(p) if os.path.isfile(p) else None


# =============================================================================
# 1) Stacked per-phase per-step (target | draft) with distinct colors + hatching
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=False)
for ax, proc in zip(axes, ["target", "draft"]):
    labels_set = set()
    per_cfg = {}
    for label, path in CONFIGS:
        df = _load(path)
        if df is None:
            per_cfg[label] = pd.Series(dtype=float)
            continue
        sub = df[df["proc"] == proc].set_index("label")["ms_per_step"]
        per_cfg[label] = sub
        labels_set.update(sub.index)

    # Order: by descending mean-across-runs so big bars stack at bottom.
    score = {lb: max(per_cfg[c].get(lb, 0.0) for c, _ in CONFIGS) for lb in labels_set}
    order = sorted(labels_set, key=lambda lb: -score[lb])

    bottoms = [0.0] * len(CONFIGS)
    handles = []
    for lb in order:
        vals = [per_cfg[label].get(lb, 0.0) for label, _ in CONFIGS]
        color, hatch = _style(lb)
        bars = ax.bar(range(len(CONFIGS)), vals, bottom=bottoms,
                      color=color, label=lb,
                      edgecolor="black", linewidth=0.5, hatch=hatch)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
        handles.append(bars)
    for i, b in enumerate(bottoms):
        ax.text(i, b + 0.6, f"{b:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([lbl for lbl, _ in CONFIGS], fontsize=10)
    ax.set_ylabel("ms / spec step", fontsize=11)
    ax.set_title(f"{proc} side", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, framealpha=0.95, ncol=1)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

plt.suptitle("3-way breakdown: pre-hybrid baseline (A) vs current split (B) vs current hybrid (C)\n"
             "layerskip-llama2-70B AWQ + TinyLlama-1.1B, 200×256 prompts",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 0.84, 0.95])
out = f"{RESULTS}/compare_3way_breakdown_distinct.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"-> {out}")


# =============================================================================
# 2) Two-axis horizontal bar: same data, easier to scan than stacked
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=False)
for ax, proc in zip(axes, ["target", "draft"]):
    labels_set = set()
    per_cfg = {}
    for label, path in CONFIGS:
        df = _load(path)
        if df is None:
            per_cfg[label] = pd.Series(dtype=float)
            continue
        sub = df[df["proc"] == proc].set_index("label")["ms_per_step"]
        per_cfg[label] = sub
        labels_set.update(sub.index)

    # Order: by max value descending (biggest at top)
    score = {lb: max(per_cfg[c].get(lb, 0.0) for c, _ in CONFIGS) for lb in labels_set}
    order = sorted(labels_set, key=lambda lb: score[lb])  # bottom-up = small first
    n = len(order)
    y = list(range(n))
    width = 0.27
    cfg_colors = ["#666666", "#fdae61", "#66c2a5"]   # gray (A), orange (B), green (C)
    for i, (lbl, _) in enumerate(CONFIGS):
        vals = [per_cfg[lbl].get(lb, 0.0) for lb in order]
        offset = (i - 1) * width
        bars = ax.barh([yy + offset for yy in y], vals, width,
                       color=cfg_colors[i], label=lbl.replace("\n", " "),
                       edgecolor="black", linewidth=0.4)
        for yy, v in zip(y, vals):
            if v >= 0.1:
                ax.text(v + 0.1, yy + offset, f"{v:.2f}",
                        va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("ms / spec step", fontsize=11)
    ax.set_title(f"{proc} side", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

plt.suptitle("3-way per-phase contribution per spec step (grouped horizontal bar)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{RESULTS}/compare_3way_grouped_hbar.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"-> {out}")


# =============================================================================
# 3) Per-phase delta with explicit signs (B-A, C-A, C-B)
# =============================================================================
all_means = {}
for label, path in CONFIGS:
    df = _load(path)
    if df is None:
        continue
    s = df.groupby(["proc", "label"])["mean_ms"].mean()
    all_means[label] = s

a_label, b_label, c_label = [c[0] for c in CONFIGS]
keys = sorted(set().union(*[s.index for s in all_means.values()]))
rows = []
for proc, lb in keys:
    a = all_means[a_label].get((proc, lb), float("nan")) if a_label in all_means else float("nan")
    b = all_means[b_label].get((proc, lb), float("nan")) if b_label in all_means else float("nan")
    c = all_means[c_label].get((proc, lb), float("nan")) if c_label in all_means else float("nan")
    rows.append({
        "proc": proc, "label": lb,
        "A": a, "B": b, "C": c,
        "B-A": b - a, "C-A": c - a, "C-B": c - b,
    })
deltas = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, proc in zip(axes, ["target", "draft"]):
    sub = deltas[deltas["proc"] == proc].copy()
    sub["max_abs"] = sub[["B-A", "C-A", "C-B"]].abs().max(axis=1)
    sub = sub.sort_values("max_abs", ascending=True)
    sub = sub[sub["max_abs"] > 0.01]
    n = len(sub)
    y = list(range(n))
    w = 0.27

    delta_cols = [("B-A", "#d73027", "(B−A) split regression\nfrom hybrid restructure"),
                  ("C-A", "#4575b4", "(C−A) net hybrid effect\nvs pre-hybrid"),
                  ("C-B", "#1a9850", "(C−B) hybrid vs split\non same codebase")]
    for i, (col, color, leg) in enumerate(delta_cols):
        offset = (i - 1) * w
        ax.barh([yy + offset for yy in y], sub[col], w,
                label=leg, color=color, edgecolor="black", linewidth=0.4)
        for yy, v in zip(y, sub[col]):
            if abs(v) >= 0.05:
                ha = "left" if v > 0 else "right"
                pad = 0.05 if v > 0 else -0.05
                ax.text(v + pad, yy + offset, f"{v:+.2f}",
                        va="center", ha=ha, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=9)
    ax.set_xlabel("Δ mean event time (ms)  ←faster | slower→", fontsize=11)
    ax.set_title(f"{proc} side — pairwise deltas", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle=":")

plt.suptitle("Pairwise per-phase deltas (negative = faster). Mean event time, post-warmup.",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{RESULTS}/compare_3way_phase_delta_distinct.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"-> {out}")


# =============================================================================
# 4) Top-line summary with distinct colors
# =============================================================================
PATTERNS = {
    "TPS (tok/s)":   r"Total Throughput:\s*([\d.]+)",
    "accept":        r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)",
    "cache_hits":    r"Avg Cache Hits:\s*([\d.]+)",
    "P1_hit":        r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)",
    "P2_hit":        r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)",
    "draft_ms":      r"Avg draft step time \(ms\):\s*([\d.]+)",
    "verify_ms":     r"Avg target verify time \(ms\):\s*([\d.]+)",
}


def _extract(path):
    log = f"{path}/run.log"
    if not os.path.isfile(log):
        return {k: None for k in PATTERNS}
    text = open(log).read()
    return {k: float(m.group(1)) if (m := re.search(p, text)) else None
            for k, p in PATTERNS.items()}


metrics = {label: _extract(path) for label, path in CONFIGS}
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
labels = [c[0] for c in CONFIGS]
short_labels = ["A: pre-hybrid", "B: current split", "C: current hybrid"]
colors = ["#4d4d4d", "#d73027", "#1a9850"]
for ax, key in zip(axes.flatten(), PATTERNS.keys()):
    vals = [metrics[lbl][key] for lbl in labels]
    bars = ax.bar(short_labels, [v if v is not None else 0 for v in vals],
                  color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(key, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
axes.flatten()[-1].axis("off")
plt.suptitle("Top-line metrics: A (pre-hybrid baseline) vs B (split fallback) vs C (hybrid)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{RESULTS}/compare_3way_summary_distinct.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"-> {out}")


# =============================================================================
# 5) Color legend reference card
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 9))
phases_in_order = list(PHASE_STYLE.keys())
n = len(phases_in_order)
y = list(range(n))
for i, lb in enumerate(phases_in_order):
    color, hatch = PHASE_STYLE[lb]
    ax.barh([n - 1 - i], [1.0], color=color, edgecolor="black", linewidth=0.5, hatch=hatch)
    ax.text(1.05, n - 1 - i, lb, va="center", fontsize=10)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
ax.set_title("Phase color/hatch legend reference", fontsize=12)
plt.tight_layout()
out = f"{RESULTS}/phase_legend.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"-> {out}")
