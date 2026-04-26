"""Stage 5 confirmation comparison: best async vs best MESA, side-by-side
phase breakdown for both target and draft processes."""
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/chokwans99/PSD/ssd/experiments/sweep_70b_awq")

CONFIGS = [
    ("stage5/modeasync_k7_f8_ns200_ol256",                     "Async\nk=7 f=8"),
    ("stage5/modemesa_k6_f3_dfo2_exit53_policya_ns200_ol256",  "MESA\nk=6 f=3 dfo=2\nexit=53 policy=A"),
]

COLORS = {
    "glue": "#17becf", "phase1_build": "#a9cce3", "phase1_prep": "#807dba",
    "phase1_replay": "#3f007d", "phase2_build": "#c7e9c0", "phase2_prep": "#74c476",
    "phase2_replay": "#00441b", "proxy_wait": "#ff7f00", "merge_cache": "#252525",
    "tree_prep": "#fcae91", "tree_replay": "#cb181d", "draft_recv_cmd": "#bdbdbd",
    "hit_cache_respond": "#78c679", "draft_send_response": "#006d2c",
    "draft_glue_replay": "#1f78b4", "verify_setup": "#d9d9d9", "graph_pre": "#e41a1c",
    "exit_logits": "#bdbdbd", "proxy_compute_send": "#ffbf00", "graph_post": "#a50f15",
    "final_logits": "#737373", "verify_sample_accept": "#fc9272",
    "target_spec_wait": "#fee090", "target_postprocess": "#fdae61",
    "verify_replay": "#b30000",
}


def throughput(log_path):
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.search(r"Total Throughput:\s*([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)

for ax, proc in zip(axes, ("target", "draft")):
    labels_set = set()
    for cfg, _ in CONFIGS:
        csv = ROOT / cfg / "mesa_per_step_contribution.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        labels_set.update(df[df["proc"] == proc]["label"].tolist())

    seed_csv = ROOT / CONFIGS[0][0] / "mesa_per_step_contribution.csv"
    if seed_csv.exists():
        seed = pd.read_csv(seed_csv)
        order = list(seed[seed["proc"] == proc]
                     .sort_values("ms_per_step", ascending=False)["label"])
    else:
        order = sorted(labels_set)
    for lb in labels_set:
        if lb not in order:
            order.append(lb)

    n = len(CONFIGS)
    w = 0.5
    xs = list(range(n))
    seen_labels = set()
    for i, (cfg, _) in enumerate(CONFIGS):
        csv = ROOT / cfg / "mesa_per_step_contribution.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        sub = df[df["proc"] == proc].set_index("label")["ms_per_step"]
        bottom = 0.0
        for lb in order:
            v = float(sub.get(lb, 0.0))
            if v <= 0:
                continue
            label_arg = lb if lb not in seen_labels else None
            ax.bar(i, v, w, bottom=bottom,
                   color=COLORS.get(lb, "#bdc3c7"),
                   edgecolor="black", linewidth=0.3,
                   label=label_arg)
            bottom += v
            seen_labels.add(lb)
        ax.text(i, bottom + 1, f"{bottom:.1f}ms", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in CONFIGS], fontsize=10)
    ax.set_ylabel("ms / spec step")
    ax.set_title(f"{proc} side")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, framealpha=0.95)

# Title with TPs
tps = []
for cfg, lbl in CONFIGS:
    tp = throughput(ROOT / cfg / "run.log")
    tps.append(f"{lbl.replace(chr(10), ' ')}: TP={tp:.2f}" if tp else lbl)
fig.suptitle(
    "Stage 5 Confirmation — 70B AWQ + TinyLlama AWQ (NS=200, OL=256)\n"
    + "  |  ".join(tps),
    fontsize=11, y=1.0,
)
plt.tight_layout(rect=[0, 0, 0.85, 0.96])
out = ROOT / "stage5_compare_breakdown.png"
plt.savefig(out, dpi=140)
print(f"-> {out}")
plt.close(fig)
