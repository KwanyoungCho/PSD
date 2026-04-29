"""Plot a compact dashboard from SSD experiment summary_index.csv."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("total_tps", "Total TPS", "tok/s", True),
    ("avg_tokens_per_step", "Tokens / Step", "tokens", True),
    ("accept_fraction", "Accept Fraction", "", True),
    ("cache_hit_rate", "Cache Hit Rate", "", True),
    ("target_verify_ms", "Target Verify", "ms", False),
    ("draft_step_ms", "Draft Step", "ms", False),
]


def _short_label(run_name: str) -> str:
    k = re.search(r"_k(\d+)_", run_name)
    f = re.search(r"_f(\d+)_", run_name)
    temp = re.search(r"_temp(\d+)", run_name)
    parts = []
    if k:
        parts.append(f"k{k.group(1)}")
    if f:
        parts.append(f"f{f.group(1)}")
    if temp:
        t = temp.group(1)
        if len(t) >= 2:
            parts.append(f"t={t[0]}.{t[1:]}")
        else:
            parts.append(f"t={t}")
    return " ".join(parts) if parts else run_name


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_summary(csv_path: Path, latest: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run_name" not in df.columns:
        raise ValueError(f"{csv_path} must contain a run_name column")
    df = df.drop_duplicates(subset=["run_name"], keep="last").reset_index(drop=True)
    if latest is not None and latest > 0:
        df = df.tail(latest).reset_index(drop=True)
    df["label"] = df["run_name"].map(_short_label)
    for col, *_ in METRICS:
        if col in df.columns:
            df[col] = _as_float(df[col])
    return df


def plot_dashboard(df: pd.DataFrame, out_path: Path, title: str | None = None) -> None:
    labels = df["label"].tolist()
    n = len(df)
    palette = ["#2f6f73", "#c65f2e", "#7b5aa6", "#3f7f3f", "#b44e5a", "#446caa"]
    colors = [palette[i % len(palette)] for i in range(n)]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    axes = axes.flatten()

    for ax, (col, name, unit, higher_is_better) in zip(axes, METRICS):
        if col not in df.columns:
            ax.set_axis_off()
            continue
        values = df[col]
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(name)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

        finite = values.dropna()
        best_idx = None
        if not finite.empty:
            best_val = finite.max() if higher_is_better else finite.min()
            best_idx = values[values == best_val].index[0]
        for i, (bar, val) in enumerate(zip(bars, values)):
            if pd.isna(val):
                continue
            height = bar.get_height()
            if abs(val) < 1:
                label = f"{val:.2f}"
            elif abs(val) < 10:
                label = f"{val:.2f}"
            else:
                label = f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == best_idx else "normal",
            )
            if i == best_idx:
                bar.set_linewidth(2.0)

        if n >= 2 and values.iloc[0] not in (None, 0) and not pd.isna(values.iloc[0]):
            base = values.iloc[0]
            for i, val in enumerate(values):
                if i == 0 or pd.isna(val):
                    continue
                delta = (val / base - 1.0) * 100.0
                sign = "+" if delta >= 0 else ""
                y = val * 0.55 if val > 0 else val
                ax.text(
                    i,
                    y,
                    f"{sign}{delta:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    if title is None:
        title = "SSD Experiment Summary"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_index", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--latest", type=int, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    out = args.out or args.summary_index.with_name("summary_dashboard.png")
    df = load_summary(args.summary_index, latest=args.latest)
    plot_dashboard(df, out, title=args.title)
    print(f"-> saved {out}")


if __name__ == "__main__":
    main()
