#!/usr/bin/env python3
"""Plot a compact, paper-oriented P1/P2 tree timeline from DUET profiles.

The detailed aligned plot is useful for debugging but contains dozens of
kernel labels.  This view groups those events into algorithmic stages while
retaining their measured wall-clock boundaries.  Multi-prompt traces are
keyed by ``(request_epoch, step_id)`` so repeated step IDs are never merged.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


SSD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSD_ROOT / "bench"))

from plot_duet_aligned_timeline import (  # noqa: E402
    _load_json,
    _strip_anchor,
    compute_causality_shift_ns,
    pick_representative_occurrence,
    select_step,
    step_full_duration_ms,
    tag_request_epochs,
)


STAGES = {
    "target": [
        ("Cache wait", ("target_spec_wait",), "#4C78A8"),
        ("Verify setup", ("verify_setup",), "#A0A0A0"),
        ("Target pre-exit", ("graph_pre",), "#B64E4E"),
        ("Proxy compute/send", ("exit_proxy_side",), "#F2A65A"),
        ("Target post-exit", ("graph_post",), "#8172B3"),
        (
            "Target verify",
            ("final_logits", "verify_sample_accept", "target_postprocess"),
            "#E76F51",
        ),
    ],
    "draft": [
        (
            "Proposal response",
            (
                "draft_recv_request",
                "hit_cache_respond_hit_k1",
                "hit_cache_respond_hit_k2",
                "hit_cache_respond_miss",
                "draft_send_response",
            ),
            "#595959",
        ),
        ("Glue", ("glue",), "#3A9D8F"),
        (
            "P1 tree",
            (
                "phase1_build",
                "p1_root_build",
                "p1_slot_prepare",
                "p1_prepare",
                "p1_graph_replay",
            ),
            "#4C9BD3",
        ),
        ("Proxy wait", ("proxy_wait",), "#F2C94C"),
        (
            "P2 tree",
            (
                "phase2_build",
                "p2_prepare",
                "p2_graph_replay",
                "p2_output_convert",
                "p2_cache_merge",
                "merge_cache",
            ),
            "#2E8B57",
        ),
    ],
}


def _interval(rows: list[dict], labels: tuple[str, ...]) -> tuple[int, int] | None:
    selected = [row for row in rows if str(row.get("label", "")) in labels]
    starts = [int(row["wall_start_ns"]) for row in selected if row.get("wall_start_ns")]
    ends = [int(row["wall_end_ns"]) for row in selected if row.get("wall_end_ns")]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _draw_stage(
    ax,
    *,
    start_ns: int,
    end_ns: int,
    origin_ns: int,
    shift_ns: int,
    y: float,
    label: str,
    color: str,
) -> tuple[float, float]:
    start_ms = (start_ns - shift_ns - origin_ns) / 1e6
    end_ms = (end_ns - shift_ns - origin_ns) / 1e6
    width_ms = max(end_ms - start_ms, 0.03)
    ax.barh(
        y,
        width_ms,
        left=start_ms,
        height=0.56,
        color=color,
        edgecolor="white",
        linewidth=1.1,
        zorder=2,
    )
    # Avoid clipped text in short segments; the complete mapping remains in
    # the legend.  Long labels need proportionally more room.
    min_label_width = max(4.3, 0.68 * len(label))
    if width_ms >= min_label_width:
        ax.text(
            start_ms + width_ms / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
            color="white" if color not in {"#F2C94C", "#A0A0A0"} else "#222",
            clip_on=True,
            zorder=3,
        )
    return start_ms, end_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument(
        "--skip-request-epochs",
        type=int,
        default=0,
        help="exclude initial warmup generate() calls from representative selection",
    )
    parser.add_argument(
        "--output-stem",
        default="timeline_p1_p2_tree_overview",
        help="PNG/PDF/CSV output basename",
    )
    args = parser.parse_args()

    target = tag_request_epochs(_strip_anchor(_load_json(args.profile_dir, "target_rank0")))
    draft = tag_request_epochs(_strip_anchor(_load_json(args.profile_dir, "draft")))
    if args.skip_request_epochs:
        target = [
            row for row in target
            if int(row["_request_epoch"]) >= args.skip_request_epochs
        ]
        draft = [
            row for row in draft
            if int(row["_request_epoch"]) >= args.skip_request_epochs
        ]

    statuses = (
        ("hit_k1", "P1 cache hit"),
        ("hit_k2", "P2 cache hit"),
        ("miss", "Cache miss"),
    )
    selected: list[dict] = []
    max_end_ms = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 8.8), sharex=True)
    for ax, (status, panel_title) in zip(axes, statuses):
        occurrence = pick_representative_occurrence(target, status, draft)
        if occurrence is None:
            raise SystemExit(f"no complete occurrence for status={status}")
        epoch, step_id = occurrence
        target_step, draft_step = select_step(
            target,
            draft,
            step_id,
            status_filter=status,
            request_epoch=epoch,
        )
        target_epoch = [row for row in target if row["_request_epoch"] == epoch]
        draft_epoch = [row for row in draft if row["_request_epoch"] == epoch]
        shift_ns, shift_pairs = compute_causality_shift_ns(
            target_epoch, draft_epoch, step_id, window=5
        )
        starts = [
            int(row["wall_start_ns"])
            for row in target_step + draft_step
            if row.get("wall_start_ns") is not None
        ]
        origin_ns = min(starts)

        manifest_row = {
            "status": status,
            "request_epoch": epoch,
            "step_id": step_id,
            "target_step_ms": f"{step_full_duration_ms(target_step):.6f}",
            "draft_causality_shift_ms": f"{shift_ns / 1e6:.6f}",
            "shift_pairs": shift_pairs,
        }

        for role, rows, y, role_shift in (
            ("target", target_step, 1.0, 0),
            ("draft", draft_step, 0.0, shift_ns),
        ):
            for stage_label, event_labels, color in STAGES[role]:
                interval = _interval(rows, event_labels)
                if interval is None:
                    continue
                start_ms, end_ms = _draw_stage(
                    ax,
                    start_ns=interval[0],
                    end_ns=interval[1],
                    origin_ns=origin_ns,
                    shift_ns=role_shift,
                    y=y,
                    label=stage_label,
                    color=color,
                )
                key = stage_label.lower().replace(" ", "_").replace("/", "_")
                manifest_row[f"{key}_start_ms"] = f"{start_ms:.6f}"
                manifest_row[f"{key}_end_ms"] = f"{end_ms:.6f}"
                max_end_ms = max(max_end_ms, end_ms)

        selected.append(manifest_row)
        ax.set_yticks([0.0, 1.0], labels=["Draft", "Target"])
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", color="#D0D0D0", linestyle=":", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.55, 1.55)
        ax.set_title(
            panel_title,
            fontsize=14,
            fontweight="semibold",
            loc="left",
        )
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Time from step start (ms)", fontsize=14, fontweight="semibold")
    axes[-1].set_xlim(-0.8, max_end_ms + 1.0)
    fig.suptitle(
        "DUET P1–P2 Tree Timeline  (K1=8, K2=4, Exit Layer=56)",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    legend_entries: list[tuple[str, str]] = []
    for role in ("target", "draft"):
        legend_entries.extend((label, color) for label, _, color in STAGES[role])
    handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in legend_entries
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.15, hspace=0.48)

    stem = args.profile_dir / args.output_stem
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fieldnames: list[str] = []
    for row in selected:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
