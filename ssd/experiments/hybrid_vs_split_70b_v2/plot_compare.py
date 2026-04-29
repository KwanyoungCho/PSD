"""Plot hybrid vs optimized-split (NEW) comparison for T10 70B sweep.

Predictor-driven candidates: 5 (dfo, pfo) combos × 2 modes.
"""
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

CONFIGS = [
    ("dfo=2,pfo=3\nK1=9,K2=1", "dfo2_pfo3_K10_K1_9_K2_1"),
    ("dfo=3,pfo=3\nK1=8,K2=2", "dfo3_pfo3_K10_K1_8_K2_2"),
    ("dfo=3,pfo=5\nK1=8,K2=2", "dfo3_pfo5_K10_K1_8_K2_2"),
    ("dfo=4,pfo=4\nK1=7,K2=2", "dfo4_pfo4_K9_K1_7_K2_2"),
    ("dfo=4,pfo=6\nK1=7,K2=2", "dfo4_pfo6_K9_K1_7_K2_2"),
]

PATTERNS = {
    "tps": re.compile(r"Total Throughput:\s*([\d.]+)"),
    "accept": re.compile(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)"),
    "p1": re.compile(r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)"),
    "p2": re.compile(r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)"),
    "draft_ms": re.compile(r"Avg draft step time \(ms\):\s*([\d.]+)"),
    "verify_ms": re.compile(r"Avg target verify time \(ms\):\s*([\d.]+)"),
}


def parse_log(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        text = f.read()
    out = {}
    for k, p in PATTERNS.items():
        m = p.search(text)
        out[k] = float(m.group(1)) if m else None
    return out


def parse_breakdown(d):
    csv_path = os.path.join(d, "mesa_breakdown_summary.csv")
    if not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path) as f:
        for ln in f.readlines()[1:]:
            parts = ln.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                proc, label, mean_ms = parts[0], parts[1], float(parts[2])
            except ValueError:
                continue
            out[f"{proc}:{label}"] = mean_ms
    return out


def main():
    rows = []
    for label, base in CONFIGS:
        h = parse_log(os.path.join(RES, base + "_hybrid", "run.log"))
        s = parse_log(os.path.join(RES, base + "_split", "run.log"))
        h_bd = parse_breakdown(os.path.join(RES, base + "_hybrid"))
        s_bd = parse_breakdown(os.path.join(RES, base + "_split"))
        rows.append((label, h, s, h_bd, s_bd))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(rows))
    w = 0.38

    h_tps = [r[1].get("tps") or 0 for r in rows]
    s_tps = [r[2].get("tps") or 0 for r in rows]
    ax = axes[0]
    bh = ax.bar(x - w / 2, h_tps, w, label="hybrid", color="#3b82f6")
    bs = ax.bar(x + w / 2, s_tps, w, label="split (NEW)", color="#f97316")
    for b, v in zip(bh, h_tps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    for b, v in zip(bs, s_tps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Total Throughput (tokens/s)")
    ax.set_title("TPS — Hybrid vs Split (NEW)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    h_acc = [r[1].get("accept") or 0 for r in rows]
    s_acc = [r[2].get("accept") or 0 for r in rows]
    ax = axes[1]
    bh = ax.bar(x - w / 2, h_acc, w, label="hybrid", color="#3b82f6")
    bs = ax.bar(x + w / 2, s_acc, w, label="split (NEW)", color="#f97316")
    for b, v in zip(bh, h_acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    for b, v in zip(bs, s_acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Accept rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    h_d = [r[1].get("draft_ms") or 0 for r in rows]
    s_d = [r[2].get("draft_ms") or 0 for r in rows]
    h_v = [r[1].get("verify_ms") or 0 for r in rows]
    s_v = [r[2].get("verify_ms") or 0 for r in rows]
    ax = axes[2]
    ax.bar(x - w / 2, h_d, w, label="hybrid draft", color="#3b82f6")
    ax.bar(x - w / 2, h_v, w, bottom=h_d, label="hybrid verify", color="#93c5fd")
    ax.bar(x + w / 2, s_d, w, label="split draft", color="#f97316")
    ax.bar(x + w / 2, s_v, w, bottom=s_d, label="split verify", color="#fdba74")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Step time (ms)")
    ax.set_title("Draft + Verify step time")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "MESA Phase 2: Hybrid vs Optimized Split (NEW) — 70B both-AWQ, predictor-driven (K1, K2)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = os.path.join(RES, "compare_hybrid_vs_split.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
