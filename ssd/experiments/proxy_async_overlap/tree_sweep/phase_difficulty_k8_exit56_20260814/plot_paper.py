#!/usr/bin/env python3
"""Create paper-ready phase difficulty and AL-survival figures."""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ORDER = ("P1", "P2", "Miss (fresh JIT)")
DISPLAY = {
    "P1": "Draft-source hit",
    "P2": "Proxy-source hit",
    "Miss (fresh JIT)": "Cache miss\n(Re-draft)",
}
COLORS = {"chain": "#4C78A8", "tree": "#E45756"}
SOURCE_COLORS = {"P1": "#4C78A8", "P2": "#E45756",
                 "Miss (fresh JIT)": "#72B7B2"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cluster_stat(rows, field, seed=20260814, nboot=5000):
    byq = defaultdict(list)
    for r in rows:
        byq[r["question_key"]].append(float(r[field]))
    vals = [statistics.fmean(v) for v in byq.values()]
    mean = statistics.fmean(vals)
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(vals, k=len(vals)))
                   for _ in range(nboot))
    return mean, boots[int(.025*nboot)], boots[int(.975*nboot)]


def paired_fresh_contrast(rows, field, left, right, seed=20260814,
                          nboot=5000):
    bysq = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bysq[row["source"]][row["question_key"]].append(float(row[field]))
    shared = set(bysq[left]) & set(bysq[right])
    vals = [statistics.fmean(bysq[left][q])
            - statistics.fmean(bysq[right][q]) for q in shared]
    mean = statistics.fmean(vals)
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(vals, k=len(vals)))
                   for _ in range(nboot))
    return mean, boots[int(.025*nboot)], boots[int(.975*nboot)], len(vals), \
        sum(value < 0 for value in vals) / len(vals)


def style(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.0, alpha=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="analysis")
    args = ap.parse_args()
    root = Path(args.analysis)
    summary = read_csv(root / "phase_summary.csv")
    events = read_csv(root / "phase_events.csv")
    fresh_path = root / "fresh_draft_events.csv"
    fresh = read_csv(fresh_path) if fresh_path.exists() else []
    arms = sorted({r["arm"] for r in summary},
                  key=lambda x: ("chain" not in x.lower(), x))
    arm_kind = {a: ("tree" if "tree" in a.lower() else "chain") for a in arms}

    plt.rcParams.update({
        "font.size": 19, "axes.labelsize": 23, "axes.titlesize": 25,
        "xtick.labelsize": 20, "ytick.labelsize": 20,
        "legend.fontsize": 20, "axes.labelweight": "bold",
        "axes.titleweight": "bold", "font.weight": "medium",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    if fresh:
        fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.0), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(9.0, 6.0), constrained_layout=True)
        axes = [axes]
    x = np.arange(len(ORDER)); width = .34
    for ai, arm in enumerate(arms):
        rows = {r["source"]: r for r in summary if r["arm"] == arm}
        means = [float(rows[s]["question_mean_al"]) for s in ORDER]
        lo = [means[i] - float(rows[s]["ci95_low"]) for i, s in enumerate(ORDER)]
        hi = [float(rows[s]["ci95_high"]) - means[i] for i, s in enumerate(ORDER)]
        pos = x + (ai - (len(arms)-1)/2) * width
        bar_colors = ([SOURCE_COLORS[source] for source in ORDER]
                      if len(arms) == 1 else COLORS[arm_kind[arm]])
        axes[0].bar(pos, means, width*.92, label=arm,
                    color=bar_colors, edgecolor="white", linewidth=1.0)
        axes[0].errorbar(pos, means, yerr=[lo, hi], fmt="none", ecolor="#222222",
                         elinewidth=1.6, capsize=4, capthick=1.6)
    axes[0].set_xticks(x, [DISPLAY[source] for source in ORDER])
    axes[0].set_ylabel("Accepted length (tokens)")
    axes[0].set_title("(a) Observed Accepted Length")
    if len(arms) > 1:
        axes[0].legend(frameon=False, loc="upper right")
    style(axes[0])

    if fresh:
        fresh_summary = []
        for arm in arms:
            for source in ORDER:
                rows = [r for r in fresh if r["arm"] == arm and r["source"] == source]
                m, lo, hi = cluster_stat(rows, "fresh_greedy_al")
                nll, nll_lo, nll_hi = cluster_stat(rows, "fresh_first_token_nll")
                rank, rank_lo, rank_hi = cluster_stat(rows, "fresh_first_token_rank")
                fresh_summary.append({"arm": arm, "source": source,
                                      "question_mean_fresh_al": m,
                                      "ci95_low": lo, "ci95_high": hi,
                                      "question_mean_first_nll": nll,
                                      "nll_ci95_low": nll_lo,
                                      "nll_ci95_high": nll_hi,
                                      "question_mean_first_rank": rank,
                                      "rank_ci95_low": rank_lo,
                                      "rank_ci95_high": rank_hi,
                                      "events": len(rows)})
        with (root / "fresh_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fresh_summary[0]))
            w.writeheader(); w.writerows(fresh_summary)
        fresh_by_subtask = []
        for arm in arms:
            for source in ORDER:
                for group in sorted({r["group"] for r in fresh}):
                    rows = [r for r in fresh if r["arm"] == arm
                            and r["source"] == source and r["group"] == group]
                    m, lo, hi = cluster_stat(rows, "fresh_greedy_al")
                    fresh_by_subtask.append({
                        "arm": arm, "source": source, "group": group,
                        "question_mean_fresh_al": m,
                        "ci95_low": lo, "ci95_high": hi,
                        "events": len(rows),
                    })
        with (root / "fresh_by_subtask.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fresh_by_subtask[0]))
            w.writeheader(); w.writerows(fresh_by_subtask)
        fresh_contrasts = []
        for arm in arms:
            arm_rows = [r for r in fresh if r["arm"] == arm]
            for fi, field in enumerate(("fresh_greedy_al",
                                        "fresh_first_token_nll",
                                        "fresh_first_token_rank")):
                for ci, (left, right) in enumerate((
                        ("P2", "P1"), ("Miss (fresh JIT)", "P1"),
                        ("P2", "Miss (fresh JIT)"))):
                    m, lo, hi, nq, frac = paired_fresh_contrast(
                        arm_rows, field, left, right,
                        seed=20260814 + 10 * fi + ci)
                    fresh_contrasts.append({
                        "arm": arm, "metric": field,
                        "contrast": f"{left} - {right}",
                        "questions": nq, "mean_difference": m,
                        "ci95_low": lo, "ci95_high": hi,
                        "fraction_questions_below_zero": frac,
                    })
        with (root / "fresh_contrasts.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fresh_contrasts[0]))
            w.writeheader(); w.writerows(fresh_contrasts)
        for ai, arm in enumerate(arms):
            rows = {r["source"]: r for r in fresh_summary if r["arm"] == arm}
            means = [rows[s]["question_mean_fresh_al"] for s in ORDER]
            lo = [means[i] - rows[s]["ci95_low"] for i, s in enumerate(ORDER)]
            hi = [rows[s]["ci95_high"] - means[i] for i, s in enumerate(ORDER)]
            pos = x + (ai - (len(arms)-1)/2) * width
            bar_colors = ([SOURCE_COLORS[source] for source in ORDER]
                          if len(arms) == 1 else COLORS[arm_kind[arm]])
            axes[1].bar(pos, means, width*.92, label=arm,
                        color=bar_colors, edgecolor="white", linewidth=1.0)
            axes[1].errorbar(pos, means, yerr=[lo, hi], fmt="none", ecolor="#222222",
                             elinewidth=1.6, capsize=4, capthick=1.6)
        axes[1].set_xticks(x, [DISPLAY[source] for source in ORDER])
        axes[1].set_ylabel("Re-draft agreement length (tokens)")
        axes[1].set_title("(b) Re-draft at the Same Prefixes")
        style(axes[1])

    for ext in ("png", "pdf"):
        fig.savefig(root / f"phase_difficulty.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Survival view exposes whether a mean difference comes from immediate
    # failures or from the long-acceptance tail.
    fig, axes = plt.subplots(1, len(arms), figsize=(9.2*len(arms), 6.0),
                             sharey=True, constrained_layout=True)
    if len(arms) == 1:
        axes = [axes]
    for ax, arm in zip(axes, arms):
        arm_ev = [r for r in events if r["arm"] == arm]
        for source in ORDER:
            vals = [int(r["accepted_len"]) for r in arm_ev if r["source"] == source]
            ks = list(range(1, 10))
            surv = [sum(v >= k for v in vals) / len(vals) for k in ks]
            ax.plot(ks, surv, marker="o", markersize=7, linewidth=3.0,
                    color=SOURCE_COLORS[source],
                    label=DISPLAY[source].replace("\n", " "))
        ax.set_xlabel("Accepted length (or longer)")
        ax.xaxis.label.set_fontweight("normal")
        ax.set_xticks(range(1, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        style(ax)
    axes[0].set_ylabel("Probability (%)")
    axes[0].yaxis.label.set_fontweight("normal")
    axes[-1].legend(
        frameon=True, fancybox=False, framealpha=1.0,
        facecolor="white", edgecolor="#444444", loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"phase_al_survival.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
