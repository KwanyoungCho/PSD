"""Render a compact PNG metrics table from SSD summary_index.csv."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


TABLE_COLUMNS = [
    ("label", "Config"),
    ("total_tps", "TPS"),
    ("tps_delta", "TPS\nvs base"),
    ("avg_tokens_per_step", "Tok\n/step"),
    ("accept_fraction", "Accept"),
    ("cache_hit_rate", "Cache\nhit"),
    ("p1_hit", "P1\nhit"),
    ("p2_hit", "P2\nhit"),
    ("p1_avg_accepted_len", "P1 acc\nlen"),
    ("p1_acceptance_ratio", "P1\nacc"),
    ("p2_avg_accepted_len", "P2 acc\nlen"),
    ("p2_acceptance_ratio", "P2\nacc"),
    ("prof_target_spec_wait_ms", "Target\nwait"),
    ("prof_draft_proxy_wait_ms", "Proxy\nwait"),
    ("proxy_wait_median_ms", "Proxy\nwait med"),
    ("target_verify_ms", "Target\nverify"),
    ("draft_step_ms", "Draft\nstep"),
    ("prof_draft_phase1_replay_ms", "P1\nreplay"),
    ("prof_draft_phase2_replay_ms", "P2\nreplay"),
    ("phase1_replay_median_ms", "P1 replay\nmed"),
    ("phase2_replay_median_ms", "P2 replay\nmed"),
]


def _short_label(run_name: str) -> str:
    k = re.search(r"_k(\d+)_", run_name)
    f = re.search(r"_f(\d+)_", run_name)
    parts = []
    if k:
        parts.append(f"k={k.group(1)}")
    if f:
        parts.append(f"f={f.group(1)}")
    return ", ".join(parts) if parts else run_name


def _row_value(row: pd.Series, key: str) -> str | None:
    if key not in row.index or pd.isna(row[key]):
        return None
    value = row[key]
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text else None


def _config_label(row: pd.Series) -> str:
    k1 = _row_value(row, "k1") or _row_value(row, "mesa_phase1_k")
    k2 = _row_value(row, "k2") or _row_value(row, "mesa_phase2_k")
    if k1 and k2:
        dfo = _row_value(row, "dfo") or _row_value(row, "mesa_draft_fan_out")
        pfo = _row_value(row, "pfo") or _row_value(row, "mesa_proxy_fan_out")
        exit_layer = _row_value(row, "exit_layer") or _row_value(row, "mesa_exit_layer")
        parts = [f"split K1={k1}/K2={k2}"]
        fanout = []
        if dfo:
            fanout.append(f"dfo={dfo}")
        if pfo:
            fanout.append(f"pfo={pfo}")
        if fanout:
            parts.append("/".join(fanout))
        if exit_layer:
            parts.append(f"exit={exit_layer}")
        return "\n".join(parts)
    return _short_label(str(row["run_name"]))


def _fmt(col: str, val: object) -> str:
    if pd.isna(val):
        return ""
    if col in {
        "accept_fraction",
        "cache_hit_rate",
        "p1_hit",
        "p2_hit",
        "p1_acceptance_ratio",
        "p2_acceptance_ratio",
    }:
        return f"{float(val) * 100:.1f}%"
    if col == "tps_delta":
        return f"{float(val):+.1f}%"
    if col in {
        "prof_target_spec_wait_ms",
        "prof_draft_proxy_wait_ms",
        "proxy_wait_median_ms",
        "target_verify_ms",
        "draft_step_ms",
        "prof_draft_phase1_replay_ms",
        "prof_draft_phase2_replay_ms",
        "phase1_replay_median_ms",
        "phase2_replay_median_ms",
    }:
        return f"{float(val):.2f} ms"
    if col == "total_tps":
        return f"{float(val):.2f}"
    if col in {"avg_tokens_per_step", "p1_avg_accepted_len", "p2_avg_accepted_len"}:
        return f"{float(val):.2f}"
    return str(val)


def load_table(csv_path: Path, latest: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "run_name" not in df.columns:
        raise ValueError(f"{csv_path} must contain a run_name column")
    df = df.drop_duplicates(subset=["run_name"], keep="last").reset_index(drop=True)
    if latest is not None and latest > 0:
        df = df.tail(latest).reset_index(drop=True)
    df["label"] = df.apply(_config_label, axis=1)

    numeric_cols = [c for c, _ in TABLE_COLUMNS if c not in {"label", "tps_delta"}]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "total_tps" in df.columns and len(df) > 0:
        base = df["total_tps"].iloc[0]
        df["tps_delta"] = (df["total_tps"] / base - 1.0) * 100.0 if base else 0.0
    else:
        df["tps_delta"] = 0.0
    return df


def _has_nonempty_value(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    series = df[col]
    if series.notna().any():
        return series.dropna().astype(str).str.strip().ne("").any()
    return False


def plot_table(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    drop_empty_columns: bool = False,
) -> None:
    columns = []
    for col, name in TABLE_COLUMNS:
        if col not in df.columns:
            continue
        if drop_empty_columns and col != "label" and not _has_nonempty_value(df, col):
            continue
        columns.append((col, name))
    headers = [name for _, name in columns]
    cell_text = [[_fmt(col, row[col]) for col, _ in columns] for _, row in df.iterrows()]

    nrows = max(len(cell_text), 1)
    fig_w = max(11.5, len(headers) * 1.35)
    fig_h = 1.7 + nrows * 0.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title is None:
        title = "SSD Experiment Metrics"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5 if len(headers) >= 14 else 10)
    table.scale(1.0, 1.35)

    best_tps_idx = None
    best_wait_idx = None
    if "total_tps" in df.columns and not df["total_tps"].dropna().empty:
        best_tps_idx = int(df["total_tps"].idxmax())
    wait_col = None
    for candidate in ("proxy_wait_median_ms", "prof_draft_proxy_wait_ms", "prof_target_spec_wait_ms"):
        if candidate in df.columns and not df[candidate].dropna().empty:
            wait_col = candidate
            break
    if wait_col is not None:
        best_wait_idx = int(df[wait_col].idxmin())

    col_index = {col: i for i, (col, _) in enumerate(columns)}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#2b2b2b")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#1f3a3d")
            cell.set_text_props(color="white", weight="bold")
            continue
        data_idx = r - 1
        cell.set_facecolor("#f4f2ec" if data_idx % 2 == 0 else "#ffffff")
        if best_tps_idx is not None and data_idx == best_tps_idx and c in {
            col_index.get("total_tps", -1),
            col_index.get("tps_delta", -1),
        }:
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")
        if best_wait_idx is not None and wait_col is not None and data_idx == best_wait_idx and c == col_index.get(wait_col, -1):
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_index", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--latest", type=int, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--drop-empty-columns",
        action="store_true",
        help="Hide columns whose values are entirely empty/NaN in the selected rows.",
    )
    args = parser.parse_args()

    out = args.out or args.summary_index.with_name("summary_table.png")
    df = load_table(args.summary_index, latest=args.latest)
    plot_table(df, out, title=args.title, drop_empty_columns=args.drop_empty_columns)
    print(f"-> saved {out}")


if __name__ == "__main__":
    main()
