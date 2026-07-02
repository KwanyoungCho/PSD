"""70B target-AWQ — dense draft vs AWQ draft side-by-side.

Both arms use the same AWQ-quantized 70B target. Difference is only the draft:
  arm A: TinyLlama dense fp16 (tmp/final_exp2_quant_70b/<cfg>/)
  arm B: TinyLlama AWQ W4A16   (tmp/final_exp2_quant_70b/draft_awq/<cfg>/)

Plots:
  1. compare_throughput_draft_awq.png — per-config TP, dense-draft vs AWQ-draft
  2. compare_breakdown_dense_vs_awq_draft.png — per-phase stack pairs
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


BASE_70B = "/home/chokwans99/PSD/ssd/tmp/final_exp2_quant_70b"
DENSE_DRAFT_BASE = BASE_70B
AWQ_DRAFT_BASE = f"{BASE_70B}/draft_awq"

CONFIGS = [
    ("baseline_k7_uniform",    "SSD baseline\nK=7 uniform"),
    ("baseline_k7_geo",        "SSD baseline\nK=7 geo"),
    ("duet_k5_f4_dfo2_exit40", "DUET K=5\nexit=40"),
    ("duet_k5_f4_dfo2_exit47", "DUET K=5\nexit=47"),
    ("duet_k5_f4_dfo2_exit53", "DUET K=5\nexit=53"),
]

COLORS = {
    "glue":"#17becf","phase1_build":"#a9cce3","phase1_prep":"#807dba","phase1_replay":"#3f007d",
    "phase2_build":"#c7e9c0","phase2_prep":"#74c476","phase2_replay":"#00441b","proxy_wait":"#ff7f00",
    "merge_cache":"#252525","tree_prep":"#fcae91","tree_replay":"#cb181d","draft_recv_cmd":"#bdbdbd",
    "hit_cache_respond":"#78c679","draft_send_response":"#006d2c","draft_glue_replay":"#1f78b4",
    "verify_setup":"#d9d9d9","graph_pre":"#e41a1c","exit_logits":"#bdbdbd","proxy_compute_send":"#ffbf00",
    "graph_post":"#a50f15","final_logits":"#737373","verify_sample_accept":"#fc9272",
    "target_spec_wait":"#fee090","target_postprocess":"#fdae61","verify_replay":"#b30000",
}


def _throughput(log_path):
    if not os.path.isfile(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Total Throughput:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def _breakdown(csv_path):
    if not os.path.isfile(csv_path):
        return None
    return pd.read_csv(csv_path)


def plot_throughput(out_png):
    dense_tps = [_throughput(f"{DENSE_DRAFT_BASE}/{c}/run.log") for c, _ in CONFIGS]
    awq_tps   = [_throughput(f"{AWQ_DRAFT_BASE}/{c}/run.log")   for c, _ in CONFIGS]

    x = list(range(len(CONFIGS)))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([i - w/2 for i in x], [t or 0 for t in dense_tps], w,
           label="draft DENSE fp16", color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.bar([i + w/2 for i in x], [t or 0 for t in awq_tps], w,
           label="draft AWQ W4A16", color="#dd8452", edgecolor="black", linewidth=0.5)
    for i, (d, q) in enumerate(zip(dense_tps, awq_tps)):
        if d is not None:
            ax.text(i - w/2, d + 0.7, f"{d:.1f}", ha="center", fontsize=9)
        if q is not None:
            ax.text(i + w/2, q + 0.7, f"{q:.1f}", ha="center", fontsize=9)
        if d and q:
            ax.text(i, max(d, q) + 4, f"×{q/d:.2f}", ha="center",
                    fontsize=10, color="darkred", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CONFIGS], fontsize=9)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("layerskip-llama2-70B (target AWQ, TP=4) — dense draft vs AWQ draft  (50 prompts × 256 toks)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"-> {out_png}")
    plt.close(fig)


def plot_breakdown(out_png):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

    for ax, proc in zip(axes, ("target", "draft")):
        labels_set = set()
        for cfg, _ in CONFIGS:
            for base in (DENSE_DRAFT_BASE, AWQ_DRAFT_BASE):
                df = _breakdown(f"{base}/{cfg}/duet_per_step_contribution.csv")
                if df is None:
                    continue
                labels_set.update(df[df["proc"] == proc]["label"].tolist())

        seed = _breakdown(f"{DENSE_DRAFT_BASE}/baseline_k7_geo/duet_per_step_contribution.csv")
        if seed is not None:
            order = list(seed[seed["proc"] == proc]
                         .sort_values("ms_per_step", ascending=False)["label"])
        else:
            order = sorted(labels_set)
        for lb in labels_set:
            if lb not in order:
                order.append(lb)

        n = len(CONFIGS)
        w = 0.38
        group_x = list(range(n))
        for i, (cfg, _) in enumerate(CONFIGS):
            for side, base, xoff in (("dense", DENSE_DRAFT_BASE, -w/2 - 0.02),
                                     ("AWQ",   AWQ_DRAFT_BASE,   +w/2 + 0.02)):
                df = _breakdown(f"{base}/{cfg}/duet_per_step_contribution.csv")
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
                ax.text(i + xoff, bottom + 0.7, f"{bottom:.1f}", ha="center", fontsize=7)
                ax.text(i + xoff, -1.5, side, ha="center", fontsize=8, color="gray")

        ax.set_xticks(group_x)
        ax.set_xticklabels([lbl for _, lbl in CONFIGS], fontsize=8)
        ax.set_ylabel("ms / spec step")
        ax.set_title(f"{proc} side — draft DENSE vs draft AWQ (target AWQ both)")
        ax.grid(axis="y", alpha=0.3, linestyle=":")

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
    out_dir = sys.argv[1] if len(sys.argv) > 1 else AWQ_DRAFT_BASE
    os.makedirs(out_dir, exist_ok=True)
    plot_throughput(f"{out_dir}/compare_throughput_draft_awq.png")
    plot_breakdown(f"{out_dir}/compare_breakdown_dense_vs_awq_draft.png")
