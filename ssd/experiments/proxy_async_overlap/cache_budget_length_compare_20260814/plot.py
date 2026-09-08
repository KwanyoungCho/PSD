#!/usr/bin/env python3
"""Plot K=8/9 results alone and against the canonical actual-seed-1 K=10."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
K10 = Path("/home/eslab/chokwans99/baseline/results/cache_budget_hit_k10_seed1_topk90/summary.csv")
METHODS = ("duet", "only_proxy", "geo", "uniform")
BUDGETS = tuple(range(2, 9))
LABELS = {"duet": "DUET", "only_proxy": "Only-Proxy",
          "geo": "Geo", "uniform": "Uniform"}
COLORS = {"duet": "#0072B2", "only_proxy": "#E69F00",
          "geo": "#009E73", "uniform": "#D55E00"}
MARKERS = {"duet": "o", "only_proxy": "s", "geo": "^", "uniform": "D"}
LINES = {"duet": "-", "only_proxy": "--", "geo": "-.", "uniform": ":"}


class SymmetricLineHandler(HandlerBase):
    """Draw a centered marker with equally long line segments in legends."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2
        common = {
            "color": orig_handle.get_color(),
            "linestyle": orig_handle.get_linestyle(),
            "linewidth": orig_handle.get_linewidth(),
            "transform": trans,
        }
        left = Line2D([xdescent, center_x], [center_y, center_y], **common)
        right = Line2D([center_x, xdescent + width], [center_y, center_y], **common)
        marker = Line2D(
            [center_x], [center_y], linestyle="None",
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            markerfacecolor=orig_handle.get_markerfacecolor(),
            markeredgecolor=orig_handle.get_markeredgecolor(),
            markeredgewidth=orig_handle.get_markeredgewidth(),
            color=orig_handle.get_color(), transform=trans,
        )
        return [left, right, marker]


LEGEND_KWARGS = {
    "frameon": True, "fancybox": False, "facecolor": "white",
    "edgecolor": "#BDBDBD", "handlelength": 3.0,
    "handler_map": {Line2D: SymmetricLineHandler()},
}


def load(path: Path, k: int) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["complete"] == "True"]
    if len(rows) != 28:
        raise RuntimeError(f"K={k}: expected 28 complete cells, got {len(rows)}")
    for row in rows:
        row["k"] = k
        row["avg_position_budget"] = int(row["avg_position_budget"])
        row["weighted_cache_hit_rate"] = float(row["weighted_cache_hit_rate"])
    return rows


def style(ax) -> None:
    ax.set_xlabel("Average cache budget per position")
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_xticks(BUDGETS)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)


def method_lines(ax, rows: list[dict]) -> None:
    for method in METHODS:
        points = sorted((row for row in rows if row["method"] == method),
                        key=lambda row: row["avg_position_budget"])
        ax.plot(
            [row["avg_position_budget"] for row in points],
            [100 * row["weighted_cache_hit_rate"] for row in points],
            color=COLORS[method], marker=MARKERS[method],
            linestyle=LINES[method], linewidth=2.8, markersize=8,
            markeredgecolor="white", markeredgewidth=1.0,
            label=LABELS[method],
            zorder=5 if method == "geo" else 3,
        )
        if method == "geo":
            ax.lines[-1].set_markersize(10)


def main() -> None:
    plt.rcParams.update({
        "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 14, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    by_k = {
        10: load(K10, 10),
        9: load(ROOT / "results/k9_seed1/summary.csv", 9),
        8: load(ROOT / "results/k8_seed1/summary.csv", 8),
    }
    all_values = [100 * row["weighted_cache_hit_rate"]
                  for rows in by_k.values() for row in rows]
    ymin = 5 * int((min(all_values) - 2) // 5)
    ymax = 5 * int((max(all_values) + 7) // 5)

    for k in (9, 8):
        fig, ax = plt.subplots(figsize=(8.4, 5.4))
        method_lines(ax, by_k[k])
        style(ax)
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="lower right", ncol=2, **LEGEND_KWARGS)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(ROOT / f"cache_hit_vs_budget_k{k}.{ext}",
                        dpi=320, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.7), sharex=True, sharey=True)
    for ax, k in zip(axes, (10, 9, 8)):
        method_lines(ax, by_k[k])
        style(ax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"K={k} ({k + 1} positions)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[2].legend(loc="lower right", **LEGEND_KWARGS)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"cache_hit_vs_budget_k10_k9_k8.{ext}",
                    dpi=320, bbox_inches="tight")
    plt.close(fig)

    length_colors = {10: "#4C78A8", 9: "#F58518", 8: "#54A24B"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        for k in (10, 9, 8):
            points = sorted(
                (row for row in by_k[k] if row["method"] == method),
                key=lambda row: row["avg_position_budget"])
            ax.plot(
                [row["avg_position_budget"] for row in points],
                [100 * row["weighted_cache_hit_rate"] for row in points],
                marker="o", linewidth=2.7, markersize=7,
                color=length_colors[k], label=f"K={k} ({k + 1} positions)")
        style(ax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(LABELS[method])
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    axes[1, 1].legend(loc="lower right", **LEGEND_KWARGS)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"cache_hit_vs_budget_by_method_length.{ext}",
                    dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
