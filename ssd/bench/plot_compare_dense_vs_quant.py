"""Side-by-side bar: dense final_exp2 vs AWQ final_exp2_quant.

Plots two panels:
  1. Throughput per config (AR + 5 spec configs), dense vs quant
  2. Per-phase breakdown (target + draft) for each spec config,
     dense vs quant side-by-side (pair of stacked bars per config)

Expects:
  tmp/final_exp2/<cfg>/mesa_per_step_contribution.csv
  tmp/final_exp2_quant/<cfg>/mesa_per_step_contribution.csv
  tmp/final_exp2/<cfg>/run.log       — for Throughput
  tmp/final_exp2_quant/<cfg>/run.log
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


DENSE_BASE = "/home/chokwans99/PSD/ssd/tmp/final_exp2"
QUANT_BASE = "/home/chokwans99/PSD/ssd/tmp/final_exp2_quant"

CONFIGS = [
    ("ar",                       "AR\nTP=4"),
    ("baseline_k7_uniform",      "SSD baseline\nK=7 uniform"),
    ("baseline_k7_geo",          "SSD baseline\nK=7 geo"),
    ("mesa_k5_f4_dfo2_exit24",   "MESA K=5\nexit=24"),
    ("mesa_k5_f4_dfo2_exit28",   "MESA K=5\nexit=28"),
    ("mesa_k5_f4_dfo2_exit32",   "MESA K=5\nexit=32"),
]

# Phase colors (matches plot_compare_breakdown.py)
COLORS = {
    "glue":                 "#17becf",
    "phase1_build":         "#a9cce3",
    "phase1_prep":          "#807dba",
    "phase1_replay":        "#3f007d",
    "phase2_build":         "#c7e9c0",
    "phase2_prep":          "#74c476",
    "phase2_replay":        "#00441b",
    "proxy_wait":           "#ff7f00",
    "merge_cache":          "#252525",
    "tree_prep":            "#fcae91",
    "tree_replay":          "#cb181d",
    "draft_recv_cmd":       "#bdbdbd",
    "hit_cache_respond":    "#78c679",
    "draft_send_response":  "#006d2c",
    "draft_glue_replay":    "#1f78b4",
    "verify_setup":         "#d9d9d9",
    "graph_pre":            "#e41a1c",
    "exit_logits":          "#bdbdbd",
    "proxy_compute_send":   "#ffbf00",
    "graph_post":           "#a50f15",
    "final_logits":         "#737373",
    "verify_sample_accept": "#fc9272",
    "target_spec_wait":     "#fee090",
    "target_postprocess":   "#fdae61",
    "verify_replay":        "#b30000",
}


def _throughput(log_path: str) -> float | None:
    if not os.path.isfile(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Total Throughput:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def _breakdown(csv_path: str) -> pd.DataFrame | None:
    if not os.path.isfile(csv_path):
        return None
    return pd.read_csv(csv_path)


def plot_throughput(out_png: str) -> None:
    dense_tps = [_throughput(f"{DENSE_BASE}/{c}/run.log") for c, _ in CONFIGS]
    quant_tps = [_throughput(f"{QUANT_BASE}/{c}/run.log") for c, _ in CONFIGS]

    x = list(range(len(CONFIGS)))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_d = ax.bar([i - w / 2 for i in x], [t or 0 for t in dense_tps], w,
                    label="dense fp16", color="#4c72b0", edgecolor="black", linewidth=0.5)
    bars_q = ax.bar([i + w / 2 for i in x], [t or 0 for t in quant_tps], w,
                    label="AWQ W4A16 (Marlin)", color="#dd8452", edgecolor="black", linewidth=0.5)
    for i, (d, q) in enumerate(zip(dense_tps, quant_tps)):
        if d is not None:
            ax.text(i - w / 2, d + 1, f"{d:.1f}", ha="center", fontsize=9)
        if q is not None:
            ax.text(i + w / 2, q + 1, f"{q:.1f}", ha="center", fontsize=9)
        if d and q:
            speedup = q / d
            ax.text(i, max(d, q) + 8, f"×{speedup:.2f}", ha="center",
                    fontsize=10, color="darkred", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CONFIGS], fontsize=9)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("layerskip-codellama-34B — dense fp16 vs AWQ W4A16  (200 prompts, 256 out-toks)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"-> {out_png}")
    plt.close(fig)


def plot_breakdown(out_png: str) -> None:
    """Per-config stacked bar pairs (dense | quant) for target + draft."""
    spec_configs = [c for c in CONFIGS if c[0] != "ar"]   # spec only; AR has no per-phase data
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

    for ax, proc in zip(axes, ("target", "draft")):
        # collect labels across everything
        labels_set = set()
        payload = {}
        for cfg, _ in spec_configs:
            for base in (DENSE_BASE, QUANT_BASE):
                df = _breakdown(f"{base}/{cfg}/mesa_per_step_contribution.csv")
                if df is None:
                    continue
                sub = df[df["proc"] == proc].set_index("label")["ms_per_step"]
                labels_set.update(sub.index)

        # order by dense baseline_k7_geo largest-first
        seed_df = _breakdown(f"{DENSE_BASE}/baseline_k7_geo/mesa_per_step_contribution.csv")
        if seed_df is not None:
            order = list(seed_df[seed_df["proc"] == proc].set_values(ascending=False)["label"]) \
                if False else list(seed_df[seed_df["proc"] == proc]
                                   .sort_values("ms_per_step", ascending=False)["label"])
        else:
            order = sorted(labels_set)
        for lb in labels_set:
            if lb not in order:
                order.append(lb)

        n = len(spec_configs)
        w = 0.38
        group_x = list(range(n))
        # dense bars at i-0.21, quant at i+0.21
        for i, (cfg, _) in enumerate(spec_configs):
            for side, base, xoff in (("dense", DENSE_BASE, -w / 2 - 0.02),
                                      ("quant", QUANT_BASE, w / 2 + 0.02)):
                df = _breakdown(f"{base}/{cfg}/mesa_per_step_contribution.csv")
                if df is None:
                    continue
                sub = df[df["proc"] == proc].set_index("label")["ms_per_step"]
                bottom = 0.0
                for lb in order:
                    v = float(sub.get(lb, 0.0))
                    if v <= 0:
                        continue
                    ax.bar(i + xoff, v, w, bottom=bottom,
                           color=COLORS.get(lb, "#bdc3c7"),
                           edgecolor="black", linewidth=0.25,
                           label=lb if (i == 0 and side == "dense") else None)
                    bottom += v
                # total label
                ax.text(i + xoff, bottom + 0.7, f"{bottom:.1f}", ha="center", fontsize=7)
                # d/q tag under the bar
                ax.text(i + xoff, -1.5, side, ha="center", fontsize=8, color="gray")

        ax.set_xticks(group_x)
        ax.set_xticklabels([lbl for _, lbl in spec_configs], fontsize=8)
        ax.set_ylabel("ms / spec step")
        ax.set_title(f"{proc} side — dense vs AWQ W4A16")
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        # legend — dedupe
        h, l = ax.get_legend_handles_labels()
        seen = {}
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(out_png, dpi=130)
    print(f"-> {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else QUANT_BASE
    os.makedirs(out_dir, exist_ok=True)
    plot_throughput(f"{out_dir}/compare_throughput.png")
    plot_breakdown(f"{out_dir}/compare_breakdown_dense_vs_quant.png")
