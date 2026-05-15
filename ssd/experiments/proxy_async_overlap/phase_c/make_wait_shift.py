"""Wait-shift comparison (doc 06 §6 timeline sanity).

Compares per-event medians of the same four labels across:

  A. 20260513 pre-engine-WIP 70B K1=K2=7 baseline (legacy schema)
  B. phase0b A_detail0  (post-WIP 70B K1=K2=7, legacy schema)

Both JSONs are legacy schema (no _anchor / no wall_start_ns), so per-event
cuda_ms duration is reliable and the aligned plotter is not used.  The
point of the figure is to make visible the shift of time from
``proxy_compute_send`` (parent that included CUDA-stream waiting on the
pre-WIP commit) into ``target_spec_wait_*`` after the engine WIP landed.

Run:
    python make_wait_shift.py
Output:
    wait_shift_comparison.png + RESULTS.md (printed table) in cwd
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

# Repo root reached via the script's own location.
REPO = Path(__file__).resolve().parents[3]

A_PATH = (
    REPO
    / "experiments/paper_baselines/final_experiments/"
    / "20260513_ours_k1_7_k2_7_dfo2_pfo1_exit56_temp07_seed42_ns50_in512_out512_all"
    / "mesa_profile_target_rank0_160652.json"
)
B_PATH = (
    REPO
    / "experiments/proxy_async_overlap/phase0b/A_detail0/"
    / "mesa_profile_target_rank0_124719.json"
)

LABELS = [
    "proxy_compute_send",
    "target_spec_wait_hit_k1",
    "graph_pre",
    "graph_post",
]


def medians(path: Path, warmup: int = 50) -> dict[str, float]:
    rows = json.load(path.open())
    by_label: dict[str, list[float]] = {}
    for r in rows:
        lbl = r.get("label")
        if lbl == "_anchor" or lbl is None:
            continue
        by_label.setdefault(lbl, []).append(float(r.get("ms", 0.0)))
    out: dict[str, float] = {}
    for lbl in LABELS:
        vals = by_label.get(lbl, [])
        if len(vals) <= warmup:
            continue
        out[lbl] = statistics.median(sorted(vals)[warmup:])
    return out


def main() -> None:
    a = medians(A_PATH)
    b = medians(B_PATH)

    print("Wait-shift comparison (median ms per event, warmup=50)")
    print(f"  A: {A_PATH}")
    print(f"  B: {B_PATH}")
    print()
    print(f"  {'label':<28} {'A (pre-WIP)':>14} {'B (post-WIP)':>14} {'delta':>10}")
    rows_md = []
    for lbl in LABELS:
        av = a.get(lbl)
        bv = b.get(lbl)
        delta = (bv - av) if (av is not None and bv is not None) else None
        a_str = f"{av:.3f}" if av is not None else "--"
        b_str = f"{bv:.3f}" if bv is not None else "--"
        d_str = f"{delta:+.3f}" if delta is not None else "--"
        print(f"  {lbl:<28} {a_str:>14} {b_str:>14} {d_str:>10}")
        rows_md.append((lbl, a_str, b_str, d_str))

    # Two-subplot bar chart.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    titles = [
        "A. 2026-05-13 (pre-engine-WIP)",
        "B. phase0b A_detail0 (post-engine-WIP)",
    ]
    data = [a, b]
    colors = {
        "proxy_compute_send": "#1b9e77",
        "target_spec_wait_hit_k1": "#74c476",
        "graph_pre": "#8b1a1a",
        "graph_post": "#8073ac",
    }
    for ax, title, d in zip(axes, titles, data):
        vals = [d.get(lbl, 0.0) for lbl in LABELS]
        bars = ax.bar(
            range(len(LABELS)),
            vals,
            color=[colors[lbl] for lbl in LABELS],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(
            [lbl.replace("target_spec_wait_hit_k1", "spec_wait_hit_k1") for lbl in LABELS],
            rotation=20,
            ha="right",
            fontsize=9,
        )
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.4,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("median per-event duration (ms)")
    fig.suptitle(
        "MESA wait-shift: proxy_compute_send ↓ / target_spec_wait ↑ "
        "(70B K1=K2=7, target rank0)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path(__file__).resolve().parent / "wait_shift_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"-> saved {out_path}")


if __name__ == "__main__":
    main()
