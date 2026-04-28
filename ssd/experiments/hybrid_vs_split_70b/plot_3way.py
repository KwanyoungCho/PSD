"""3-way per-phase breakdown: pre-hybrid baseline (A) vs current split (B) vs current hybrid (C).

(A) tmp/final_exp2_quant_70b/mesa_k5_f4_dfo2_exit40        — pre-hybrid codebase, split path
(B) experiments/hybrid_vs_split_70b/results/split_k5_dfo2_exit40   — current code, split fallback
(C) experiments/hybrid_vs_split_70b/results/hybrid_k5_K1_3_K2_2_exit40 — current code, hybrid

Reads mesa_per_step_contribution.csv from each (produced by plot_mesa_breakdown.py).
Writes three figures to <results>:
    1. compare_3way_breakdown.png      stacked per-phase per-step (target | draft)
    2. compare_3way_phase_delta.png    grouped bar of per-phase mean ms (B-A, C-A, C-B)
    3. compare_3way_summary.png        TPS / accept / cache / draft_ms / verify_ms grouped
"""
import os
import sys
import re

import matplotlib.pyplot as plt
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

COLORS = {
    "glue":"#17becf",
    "phase1_build":"#a9cce3","phase1_prep":"#807dba","phase1_replay":"#3f007d",
    "phase2_build":"#c7e9c0","phase2_prep":"#74c476","phase2_replay":"#00441b",
    "phase2_hybrid_build":"#fdd0a2","phase2_hybrid_replay_long":"#e6550d",
    "phase2_hybrid_replay_short":"#fdae6b","phase2_hybrid_eager_long":"#a63603",
    "phase2_hybrid_eager_short":"#fd8d3c",
    "proxy_wait":"#ff7f00","merge_cache":"#252525",
    "tree_prep":"#fcae91","tree_replay":"#cb181d","draft_recv_cmd":"#bdbdbd",
    "hit_cache_respond":"#78c679","draft_send_response":"#006d2c","draft_glue_replay":"#1f78b4",
    "verify_setup":"#d9d9d9","graph_pre":"#e41a1c","exit_logits":"#bdbdbd",
    "proxy_compute_send":"#ffbf00","graph_post":"#a50f15","final_logits":"#737373",
    "verify_sample_accept":"#fc9272","target_spec_wait":"#fee090",
    "target_postprocess":"#fdae61","verify_replay":"#b30000",
}


def _load(cfg_path):
    p = f"{cfg_path}/mesa_per_step_contribution.csv"
    return pd.read_csv(p) if os.path.isfile(p) else None


# ======================================================
# 1) Stacked per-phase per-step (target | draft)
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
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
    base_lbl = CONFIGS[0][0]
    if base_lbl in per_cfg and len(per_cfg[base_lbl]):
        order = list(per_cfg[base_lbl].sort_values(ascending=False).index)
    else:
        order = sorted(labels_set)
    for lb in labels_set:
        if lb not in order:
            order.append(lb)
    bottoms = [0.0] * len(CONFIGS)
    for lb in order:
        vals = [per_cfg[label].get(lb, 0.0) for label, _ in CONFIGS]
        ax.bar(range(len(CONFIGS)), vals, bottom=bottoms,
               color=COLORS.get(lb, "#bdc3c7"), label=lb,
               edgecolor="black", linewidth=0.3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, b in enumerate(bottoms):
        ax.text(i, b + 0.5, f"{b:.1f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([lbl for lbl, _ in CONFIGS], fontsize=9)
    ax.set_ylabel("ms / spec step")
    ax.set_title(f"{proc} side — per-phase contribution per spec step")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
plt.suptitle("3-way breakdown: pre-hybrid baseline (A) vs current split (B) vs current hybrid (C) "
             "— layerskip-llama2-70B AWQ + TinyLlama-1.1B, 200×256 prompts",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 0.88, 0.96])
out = f"{RESULTS}/compare_3way_breakdown.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"-> {out}")


# ======================================================
# 2) Per-phase delta grouped bar (B-A, C-A, C-B)
# ======================================================
all_means = {}
for label, path in CONFIGS:
    df = _load(path)
    if df is None:
        continue
    s = df.groupby(["proc", "label"])["mean_ms"].mean()
    all_means[label] = s

if len(all_means) >= 2:
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
    deltas.to_csv(f"{RESULTS}/compare_3way_means.csv", index=False)
    print(f"-> {RESULTS}/compare_3way_means.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, proc in zip(axes, ["target", "draft"]):
        sub = deltas[deltas["proc"] == proc].copy()
        # Sort by absolute max delta across {B-A, C-A, C-B}
        sub["max_abs"] = sub[["B-A", "C-A", "C-B"]].abs().max(axis=1)
        sub = sub.sort_values("max_abs", ascending=False)
        sub = sub[sub["max_abs"] > 0.01]   # drop near-zero noise
        x = range(len(sub))
        w = 0.27
        ax.bar([i - w for i in x], sub["B-A"], w, label="(B−A) split regression",
               color="#fdae61", edgecolor="black", linewidth=0.3)
        ax.bar([i for i in x], sub["C-A"], w, label="(C−A) hybrid vs pre-hybrid",
               color="#3288bd", edgecolor="black", linewidth=0.3)
        ax.bar([i + w for i in x], sub["C-B"], w, label="(C−B) hybrid effect",
               color="#66c2a5", edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Δ mean ms / event")
        ax.set_title(f"{proc} side — per-phase mean delta (negative = faster)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.suptitle("Per-phase mean event time deltas across A/B/C", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{RESULTS}/compare_3way_phase_delta.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"-> {out}")


# ======================================================
# 3) Top-line metric grouped bar (TPS / accept / cache / draft_ms / verify_ms)
# ======================================================
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

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
labels = [c[0] for c in CONFIGS]
colors = ["#666666", "#fdae61", "#66c2a5"]
for ax, key in zip(axes.flatten(), PATTERNS.keys()):
    vals = [metrics[lbl][key] for lbl in labels]
    bars = ax.bar(labels, [v if v is not None else 0 for v in vals], color=colors,
                  edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, vals):
        if v is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(key, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
# unused last axis
axes.flatten()[-1].axis("off")
plt.suptitle("Top-line metrics — A vs B vs C", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{RESULTS}/compare_3way_summary.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"-> {out}")

# Also print summary table
print()
print("=" * 80)
print("Top-line metrics summary:")
print("=" * 80)
df_metrics = pd.DataFrame(metrics).T
print(df_metrics.to_string())
df_metrics.to_csv(f"{RESULTS}/compare_3way_summary.csv")
print(f"\n-> {RESULTS}/compare_3way_summary.csv")
