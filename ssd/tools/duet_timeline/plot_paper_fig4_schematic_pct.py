#!/usr/bin/env python3
"""Reproduce the paper Figure 4 latency-breakdown schematics.

The geometry, colors, typography, and normalized intervals in this script are
recovered from ``ssd/tmp/duet_tree_timeline/paper_fig4_schematic_pct.{png,pdf}``.
The plotted values are category means normalized by the duration of one target
decoding step; displayed labels retain the rounded percentages used in the
paper figure.

``--variant chain`` reproduces the existing figure.  ``--variant tree`` reads
the aligned full-backbone tree profile and compares P2 hit against miss because
both use the same eight-node target verification bucket.  The tree figure uses
one shared target verification duration in both panels and preserves the exact
paper notation used by the chain figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# The original tightly-cropped vector canvas.  Using it directly makes both
# the PDF page and the 170-DPI PNG deterministic.
PAGE_WIDTH_PT = 1307.2821699062
PAGE_HEIGHT_PT = 757.744
PNG_DPI = 170
TREE_FONT_SIZE = 29
TREE_AXIS_FONT_SIZE = 32
TREE_PANEL_TITLE_SIZE = 42
TREE_CANVAS_SCALE = 1.25

AX_LEFT_PT = 135.684375
AX_WIDTH_PT = 1164.3977949062
AX_HEIGHT_PT = 229.36
TOP_BOTTOM_PT = 521.184
BOTTOM_BOTTOM_PT = 124.944
TREE_AX_LEFT_PT = AX_LEFT_PT + 30.0
TREE_AX_WIDTH_PT = AX_WIDTH_PT - 30.0
TREE_AX_CENTER_PT = TREE_AX_LEFT_PT + TREE_AX_WIDTH_PT / 2
TREE_BOTTOM_BOTTOM_PT = BOTTOM_BOTTOM_PT + 22.0
TREE_TOP_TITLE_Y_PT = 395.0

X_LIM = (-0.5, 103.0)
TARGET_Y = 0.704
DRAFT_Y = 0.288
BAR_HEIGHT = 0.224

COLORS = {
    "sync": "#c9c9c9",
    "verify_pre": "#a63232",
    "verify_post": "#8a6fb3",
    "sample": "#d9a3a3",
    "context": "#2ba8a0",
    "draft": "#2f6db5",
    "proxy_draft": "#19b4d1",
    "proxy": "#f28c1e",
    "redraft": "#d94040",
    "redraft_edge": "#cc2222",
    "annotation": "#b06000",
    "headroom": "#6b5d3f",
    "text": "#333333",
}

# Exact unrounded endpoints extracted from the original vector figure.
HIT = {
    "sync_end": 5.993690839958169,
    "verify_pre_end": 68.76971610290893,
    "proxy_arrival": 70.34700318009062,
    "verify_post_end": 93.84858042565752,
    "sample_end": 100.0,
    "context_end": 9.305993728705882,
    "draft_end": 47.9495268085525,
    "proxy_draft_end": 95.26813885734204,
}

MISS = {
    "sync_end": 5.093833789692993,
    "stall_end": 20.107238629490155,
    "verify_pre_end": 73.45844502329493,
    "proxy_arrival": 74.798927669704,
    "verify_post_end": 94.77211799010968,
    "sample_end": 100.0,
    "redraft_end": 15.013404839797161,
    "context_end": 22.92225205361846,
    "draft_end": 55.76407506845394,
    "proxy_draft_end": 95.97855232743424,
}


# Measured source for the tree variant.  This is the profiling companion of
# the full-backbone P1+P2 tree experiment described in TIMELINE_CONFIG.md.
DEFAULT_TREE_PROFILE_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments/proxy_async_overlap/tree_sweep/"
    "p1_tree_full_backbone_profile_20260811/p1_backbone_profile"
)

TREE_TARGET_LABELS = {
    "wait": ("target_spec_wait",),
    "setup": ("verify_setup",),
    "pre": ("graph_pre",),
    "proxy": ("exit_proxy_side",),
    "post": ("graph_post",),
    "sample": ("final_logits", "verify_sample_accept", "target_postprocess"),
}

TREE_DRAFT_LABELS = {
    "response": (
        "draft_recv_request",
        "hit_cache_respond_hit_k2",
        "hit_cache_respond_miss",
        "draft_send_response",
    ),
    "glue": ("glue",),
    "p1": (
        "phase1_build",
        "p1_root_build",
        "p1_slot_prepare",
        "p1_prepare",
        "p1_graph_replay",
    ),
    "proxy_wait": ("proxy_wait",),
    "p2": (
        "phase2_build",
        "p2_prepare",
        "p2_graph_replay",
        "p2_output_convert",
        "p2_cache_merge",
        "merge_cache",
    ),
}


def _figure_xy(x_pt: float, y_pt: float) -> tuple[float, float]:
    return x_pt / PAGE_WIDTH_PT, y_pt / PAGE_HEIGHT_PT


def _figure_text(
    fig: plt.Figure,
    x_pt: float,
    y_pt: float,
    label: str,
    *,
    color: str,
    fontsize: float = 22,
) -> None:
    """Place text by its baseline in the original PDF's point coordinates."""
    fig.text(
        *_figure_xy(x_pt, y_pt),
        label,
        ha="left",
        va="baseline",
        fontsize=fontsize,
        color=color,
    )


def _figure_line(
    fig: plt.Figure,
    x0_pt: float,
    y0_pt: float,
    x1_pt: float,
    y1_pt: float,
    *,
    color: str,
    linewidth: float = 1.6,
) -> None:
    """Draw an annotation line in PDF point coordinates."""
    fig.add_artist(
        Line2D(
            [x0_pt / PAGE_WIDTH_PT, x1_pt / PAGE_WIDTH_PT],
            [y0_pt / PAGE_HEIGHT_PT, y1_pt / PAGE_HEIGHT_PT],
            transform=fig.transFigure,
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=6,
        )
    )


def _latest_profile_json(profile_dir: Path, tag: str) -> Path:
    paths = sorted(profile_dir.glob(f"duet_profile_{tag}_*.json"))
    if not paths:
        paths = sorted(profile_dir.glob(f"duet_profile_{tag}.json"))
    if not paths:
        raise FileNotFoundError(f"no duet_profile_{tag}_*.json in {profile_dir}")
    return paths[-1]


def _tag_request_epochs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag repeated per-request step IDs without changing the profile files."""
    labels = {str(row.get("label", "")) for row in rows}
    for candidate in (
        "target_send_request",
        "draft_recv_request",
        "target_spec_wait",
        "draft_send_response",
    ):
        if candidate in labels:
            anchor = candidate
            break
    else:
        anchor = None

    epoch = 0
    previous_step: int | None = None
    tagged: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        step_id = row.get("step_id")
        if step_id is not None and (anchor is None or row.get("label") == anchor):
            step_id = int(step_id)
            if previous_step is not None and step_id <= previous_step:
                epoch += 1
            previous_step = step_id
        copy["_request_epoch"] = epoch
        tagged.append(copy)
    return tagged


def _profile_interval(
    rows: list[dict[str, Any]], labels: tuple[str, ...]
) -> tuple[int, int] | None:
    selected = [
        row
        for row in rows
        if str(row.get("label", "")) in labels
        and row.get("wall_start_ns") is not None
        and row.get("wall_end_ns") is not None
    ]
    if not selected:
        return None
    return (
        min(int(row["wall_start_ns"]) for row in selected),
        max(int(row["wall_end_ns"]) for row in selected),
    )


def _load_tree_config(profile_dir: Path) -> dict[str, Any]:
    """Load the JSONL metadata paired with the profiling experiment."""
    candidates = sorted(profile_dir.parent.glob("*.jsonl"))
    for path in candidates:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("p1_tree") == "on" and record.get("p2_tree") == "on":
                        return record
                    break
    raise FileNotFoundError(
        "tree profile needs its adjacent experiment JSONL to validate the "
        f"verification width: {profile_dir.parent}"
    )


def _tree_case_samples(
    target: list[dict[str, Any]],
    draft: list[dict[str, Any]],
    status: str,
) -> list[dict[str, float]]:
    target_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    draft_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in target:
        if row.get("step_id") is not None:
            target_by_key[(int(row["_request_epoch"]), int(row["step_id"]))].append(row)
    for row in draft:
        if row.get("step_id") is not None:
            draft_by_key[(int(row["_request_epoch"]), int(row["step_id"]))].append(row)

    samples: list[dict[str, float]] = []
    for key, target_rows in target_by_key.items():
        canonical_status = next(
            (
                row.get("status")
                for row in target_rows
                if str(row.get("label", "")).startswith("target_spec_wait")
            ),
            None,
        )
        if canonical_status != status:
            continue
        draft_rows = draft_by_key.get(key, [])
        target_intervals = {
            name: _profile_interval(target_rows, labels)
            for name, labels in TREE_TARGET_LABELS.items()
        }
        draft_intervals = {
            name: _profile_interval(draft_rows, labels)
            for name, labels in TREE_DRAFT_LABELS.items()
        }
        if not target_rows or not all(target_intervals.values()) or not all(
            draft_intervals.values()
        ):
            continue

        target_start = min(
            int(row["wall_start_ns"])
            for row in target_rows
            if row.get("wall_start_ns") is not None
        )
        setup = target_intervals["setup"]
        pre = target_intervals["pre"]
        proxy = target_intervals["proxy"]
        post = target_intervals["post"]
        sample = target_intervals["sample"]
        assert setup and pre and proxy and post and sample

        result = {
            "wait": (
                target_intervals["wait"][1] - target_intervals["wait"][0]
            )
            / 1e6,
            "prefix": (setup[0] - target_start) / 1e6,
            "pre": (pre[1] - setup[0]) / 1e6,
            "proxy": (proxy[1] - proxy[0]) / 1e6,
            "post": (post[1] - pre[1]) / 1e6,
            "sample": (sample[1] - post[1]) / 1e6,
        }
        for name, interval in draft_intervals.items():
            assert interval is not None
            result[f"draft_{name}"] = (interval[1] - interval[0]) / 1e6
        samples.append(result)
    return samples


def _case_means(samples: list[dict[str, float]], status: str) -> dict[str, float]:
    if not samples:
        raise RuntimeError(f"no complete tree profile samples for status={status}")
    return {
        key: statistics.fmean(sample[key] for sample in samples)
        for key in samples[0]
    } | {"sample_count": float(len(samples))}


def _pct(value_ms: float, total_ms: float) -> float:
    return 100.0 * value_ms / total_ms


def load_tree_figure_data(
    profile_dir: Path, skip_request_epochs: int = 2
) -> dict[str, Any]:
    """Aggregate the matched-width P2-hit/miss cases for the paper schematic.

    A miss verifies K1 proposal tokens.  A P2 tree hit verifies at most M2
    nodes in the M2 bucket.  We accept the profile only when M2 == K1, so both
    panels execute recovery + K1 target rows.  P1 hits are intentionally not
    used: M1 differs from K1 in the paper configuration.
    """
    with _latest_profile_json(profile_dir, "target_rank0").open() as handle:
        target = json.load(handle)
    with _latest_profile_json(profile_dir, "draft").open() as handle:
        draft = json.load(handle)
    target = _tag_request_epochs(
        [row for row in target if row.get("label") != "_anchor"]
    )
    draft = _tag_request_epochs(
        [row for row in draft if row.get("label") != "_anchor"]
    )
    target = [
        row for row in target if int(row["_request_epoch"]) >= skip_request_epochs
    ]
    draft = [
        row for row in draft if int(row["_request_epoch"]) >= skip_request_epochs
    ]

    config = _load_tree_config(profile_dir)
    miss_nodes = int(config.get("engine_k1", config["k1"]))
    hit_nodes = int(config["p2_verify_nodes"])
    if hit_nodes != miss_nodes:
        raise ValueError(
            "invalid paper comparison: P2 hit and miss target verification "
            f"widths differ (M2={hit_nodes}, K1={miss_nodes})"
        )

    hit_samples = _tree_case_samples(target, draft, "hit_k2")
    miss_samples = _tree_case_samples(target, draft, "miss")
    hit_mean = _case_means(hit_samples, "hit_k2")
    miss_mean = _case_means(miss_samples, "miss")

    # Use one common target compute duration in both panels.  The mean of the
    # two category means gives equal weight to hit and miss, independent of
    # their frequency in this particular trace.  Only the miss-side exposed
    # response stall changes the target critical path.
    common = {
        key: statistics.fmean((hit_mean[key], miss_mean[key]))
        for key in ("pre", "proxy", "post", "sample")
    }
    sync_ms = hit_mean["prefix"]
    # A cache hit still pays the normal request/response path.  Stall and
    # Re-draft therefore mean the *incremental* miss penalty, not the absolute
    # miss intervals.  Measure each on its own GPU clock so the figure does not
    # mix an incremental target duration with an absolute draft duration.
    # Their residual difference (~0.06 ms in this profile) is retained rather
    # than forced away; it comes from the two profiler boundaries and noise.
    target_stall_ms = miss_mean["wait"] - hit_mean["wait"]
    draft_redraft_ms = (
        miss_mean["draft_response"] - hit_mean["draft_response"]
    )
    if target_stall_ms <= 0.0 or draft_redraft_ms <= 0.0:
        raise ValueError(
            "tree profile has no positive incremental miss penalty: "
            f"target={target_stall_ms:.6f} ms, "
            f"draft={draft_redraft_ms:.6f} ms"
        )
    verify_ms = common["pre"] + common["post"] + common["sample"]

    cases: dict[str, dict[str, float]] = {}
    for panel, source, extra_stall in (
        ("hit", hit_mean, 0.0),
        ("miss", miss_mean, target_stall_ms),
    ):
        total_ms = sync_ms + extra_stall + verify_ms
        verify_start_ms = sync_ms + extra_stall
        exit_ms = verify_start_ms + common["pre"]
        proxy_arrival_ms = exit_ms + common["proxy"]
        post_end_ms = exit_ms + common["post"]
        # A hit-side cache response is part of the short synchronization
        # prefix and is intentionally not exposed as a separate paper stage.
        # A miss-side on-demand response is visible as Re-draft/Stall.
        draft_visible_prefix_ms = 0.0 if panel == "hit" else draft_redraft_ms
        cases[panel] = {
            "total_ms": total_ms,
            "sync_end": _pct(sync_ms, total_ms),
            "stall_end": _pct(sync_ms + extra_stall, total_ms),
            "verify_pre_end": _pct(exit_ms, total_ms),
            "proxy_arrival": _pct(proxy_arrival_ms, total_ms),
            "verify_post_end": _pct(post_end_ms, total_ms),
            "sample_end": 100.0,
            "draft_response_end": _pct(draft_visible_prefix_ms, total_ms),
            "glue_end": _pct(
                draft_visible_prefix_ms + source["draft_glue"], total_ms
            ),
            "p1_end": _pct(
                draft_visible_prefix_ms
                + source["draft_glue"]
                + source["draft_p1"],
                total_ms,
            ),
            "p2_end": _pct(proxy_arrival_ms + source["draft_p2"], total_ms),
            "sample_count": source["sample_count"],
        }

    return {
        "profile_dir": profile_dir,
        "config": config,
        "verify_nodes": hit_nodes,
        "verify_rows": hit_nodes + 1,
        "sync_ms": sync_ms,
        "stall_ms": target_stall_ms,
        "redraft_ms": draft_redraft_ms,
        "verify_ms": verify_ms,
        "common": common,
        "hit_mean": hit_mean,
        "miss_mean": miss_mean,
        "cases": cases,
    }


def _setup_axis(
    fig: plt.Figure,
    bottom_pt: float,
    *,
    left_pt: float = AX_LEFT_PT,
    width_pt: float = AX_WIDTH_PT,
) -> plt.Axes:
    ax = fig.add_axes(
        [
            left_pt / PAGE_WIDTH_PT,
            bottom_pt / PAGE_HEIGHT_PT,
            width_pt / PAGE_WIDTH_PT,
            AX_HEIGHT_PT / PAGE_HEIGHT_PT,
        ]
    )
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([TARGET_Y, DRAFT_Y], ["Target GPU", "Draft GPU"])
    ax.tick_params(axis="both", labelsize=22, width=0.8, length=3.5, colors="black")
    ax.grid(axis="x", color="#b0b0b0", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.8)
    return ax


def _bar(
    ax: plt.Axes,
    y: float,
    left: float,
    right: float,
    color: str,
    *,
    edgecolor: str = "white",
    linewidth: float = 0.7,
    hatch: str | None = None,
) -> None:
    ax.add_patch(
        Rectangle(
            (left, y - BAR_HEIGHT / 2),
            right - left,
            BAR_HEIGHT,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
            zorder=3,
        )
    )


def _inside(ax: plt.Axes, x: float, y: float, label: str) -> None:
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=22,
        color="white",
        zorder=5,
    )


def _vertical_marker(
    ax: plt.Axes,
    x: float,
    *,
    linewidth: float = 1.6,
    zorder: float = 2,
) -> None:
    ax.axvline(
        x,
        ymin=0.10,
        ymax=0.93,
        color=COLORS["proxy"],
        # The chain default preserves the original PDF dash pattern.  The
        # tree variant requests a heavier foreground marker for readability.
        linestyle=(0, (3.7, 1.6)),
        linewidth=linewidth,
        zorder=zorder,
    )


def _top_panel(fig: plt.Figure, ax: plt.Axes) -> None:
    h = HIT
    _vertical_marker(ax, h["proxy_arrival"])

    _bar(ax, TARGET_Y, 0, h["sync_end"], COLORS["sync"], hatch="//")
    _bar(ax, TARGET_Y, h["sync_end"], h["verify_pre_end"], COLORS["verify_pre"])
    _bar(
        ax,
        TARGET_Y,
        h["verify_pre_end"],
        h["verify_post_end"],
        COLORS["verify_post"],
    )
    _bar(ax, TARGET_Y, h["verify_post_end"], h["sample_end"], COLORS["sample"])

    _bar(ax, DRAFT_Y, 0, h["context_end"], COLORS["context"])
    _bar(ax, DRAFT_Y, h["context_end"], h["draft_end"], COLORS["draft"])
    _bar(
        ax,
        DRAFT_Y,
        h["proxy_arrival"],
        h["proxy_draft_end"],
        COLORS["proxy_draft"],
    )

    _inside(
        ax,
        (h["sync_end"] + h["verify_pre_end"]) / 2,
        TARGET_Y,
        "Verify (0–56) 63%",
    )
    _inside(
        ax,
        (h["verify_pre_end"] + h["verify_post_end"]) / 2,
        TARGET_Y,
        "Verify (57–80) 25%",
    )
    _inside(
        ax,
        (h["context_end"] + h["draft_end"]) / 2,
        DRAFT_Y,
        "Draft-source 39%",
    )
    _inside(
        ax,
        (h["proxy_arrival"] + h["proxy_draft_end"]) / 2,
        DRAFT_Y,
        "Proxy-source 25%",
    )

    # Proxy work runs on the target side stream between the two GPU tracks.
    proxy_y0, proxy_y1 = 0.4464, 0.5456
    ax.add_patch(
        Rectangle(
            (h["verify_pre_end"], proxy_y0),
            h["proxy_arrival"] - h["verify_pre_end"],
            proxy_y1 - proxy_y0,
            facecolor=COLORS["proxy"],
            edgecolor="none",
            zorder=4,
        )
    )
    _figure_line(
        fig,
        974.726703,
        634.94656,
        932.728764,
        634.94656,
        color=COLORS["annotation"],
    )
    _figure_text(
        fig,
        977.7296448408,
        629.229763125,
        "Early-exit Proxy 1.6%",
        color=COLORS["annotation"],
    )
    _figure_text(
        fig,
        781.1660876857,
        724.85568,
        "Proxy arrives",
        color=COLORS["annotation"],
    )

    arrow_y = 0.24
    ax.annotate(
        "",
        xy=(h["proxy_arrival"], arrow_y),
        xytext=(h["draft_end"], arrow_y),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#8a7d5a",
            "linewidth": 2.2,
            "mutation_scale": 10,
        },
    )
    _figure_text(
        fig,
        721.5686180446,
        589.596355125,
        "Headroom 22%",
        color=COLORS["headroom"],
    )

    _edge_annotation(
        fig, 175.024656, 708.34176, 721.851707, 127.5637186885, 730.14083625, "Sync 6%"
    )
    _edge_annotation(
        fig, 1231.729097, 708.34176, 721.852666, 1169.5259717722, 730.14083625, "Sample 6%"
    )
    _context_annotation(
        fig, 193.656724, 549.870703, 561.55136, 98.5395369104, 530.15357, "Context Align 9%"
    )

    ax.set_xlabel("Fraction of one decoding step (%)", fontsize=22, labelpad=4)
    fig.text(*_figure_xy(717.895, 409.68609375), "(a) Cache hit", ha="center", va="baseline", fontsize=26, color="black")


def _bottom_panel(fig: plt.Figure, ax: plt.Axes) -> None:
    m = MISS
    _vertical_marker(ax, m["proxy_arrival"])

    _bar(ax, TARGET_Y, 0, m["sync_end"], COLORS["sync"], hatch="//")
    _bar(
        ax,
        TARGET_Y,
        m["sync_end"],
        m["stall_end"],
        COLORS["sync"],
        edgecolor=COLORS["redraft_edge"],
        linewidth=2.0,
        hatch="//",
    )
    _bar(ax, TARGET_Y, m["stall_end"], m["verify_pre_end"], COLORS["verify_pre"])
    _bar(
        ax,
        TARGET_Y,
        m["verify_pre_end"],
        m["verify_post_end"],
        COLORS["verify_post"],
    )
    _bar(ax, TARGET_Y, m["verify_post_end"], m["sample_end"], COLORS["sample"])

    _bar(
        ax,
        DRAFT_Y,
        0,
        m["redraft_end"],
        COLORS["redraft"],
        edgecolor=COLORS["redraft_edge"],
        linewidth=2.0,
        hatch="//",
    )
    _bar(ax, DRAFT_Y, m["redraft_end"], m["context_end"], COLORS["context"])
    _bar(ax, DRAFT_Y, m["context_end"], m["draft_end"], COLORS["draft"])
    _bar(
        ax,
        DRAFT_Y,
        m["proxy_arrival"],
        m["proxy_draft_end"],
        COLORS["proxy_draft"],
    )

    _inside(
        ax,
        (m["stall_end"] + m["verify_pre_end"]) / 2,
        TARGET_Y,
        "Verify (0–56) 53%",
    )
    _inside(
        ax,
        (m["verify_pre_end"] + m["verify_post_end"]) / 2,
        TARGET_Y,
        "Verify (57–80) 21%",
    )
    _inside(ax, m["redraft_end"] / 2, DRAFT_Y, "Re-draft 15%")
    _inside(
        ax,
        (m["context_end"] + m["draft_end"]) / 2,
        DRAFT_Y,
        "Draft-source 33%",
    )
    _inside(
        ax,
        (m["proxy_arrival"] + m["proxy_draft_end"]) / 2,
        DRAFT_Y,
        "Proxy-source 21%",
    )

    stall = ax.text(
        (m["sync_end"] + m["stall_end"]) / 2,
        TARGET_Y,
        "Stall 15%",
        ha="center",
        va="center",
        fontsize=22,
        color=COLORS["text"],
        zorder=7,
    )
    stall.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])

    proxy_y0, proxy_y1 = 0.4464, 0.5456
    ax.add_patch(
        Rectangle(
            (m["verify_pre_end"], proxy_y0),
            m["proxy_arrival"] - m["verify_pre_end"],
            proxy_y1 - proxy_y0,
            facecolor=COLORS["proxy"],
            edgecolor="none",
            zorder=4,
        )
    )
    _figure_line(
        fig,
        1024.811833,
        238.70656,
        982.813895,
        238.70656,
        color=COLORS["annotation"],
    )
    _figure_text(
        fig,
        1027.814776,
        232.989763125,
        "Early-exit Proxy 1.3%",
        color=COLORS["annotation"],
    )
    _figure_text(
        fig,
        831.2512183483,
        328.61568,
        "Proxy arrives",
        color=COLORS["annotation"],
    )

    arrow_y = 0.24
    ax.annotate(
        "",
        xy=(m["proxy_arrival"], arrow_y),
        xytext=(m["draft_end"], arrow_y),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#8a7d5a",
            "linewidth": 2.2,
            "mutation_scale": 10,
        },
    )
    _figure_text(
        fig,
        790.5688780485,
        193.356355125,
        "Headroom 19%",
        color=COLORS["headroom"],
    )

    _edge_annotation(
        fig, 169.962861, 312.10176, 325.611707, 122.5019236885, 333.90083625, "Sync 5%"
    )
    _edge_annotation(
        fig, 1236.924097, 312.10176, 325.612666, 1174.7209717722, 333.90083625, "Sample 5%"
    )
    _context_annotation(
        fig, 354.701732, 153.630703, 165.31136, 259.5845449104, 133.91357, "Context Align 8%"
    )

    ax.set_xlabel("Fraction of one decoding step (%)", fontsize=22, labelpad=4)
    fig.text(*_figure_xy(717.895, 13.44609375), "(b) Cache miss", ha="center", va="baseline", fontsize=26, color="black")


def _rounded_width(left: float, right: float) -> int:
    return int(round(right - left))


def _tree_inside(
    ax: plt.Axes,
    left: float,
    right: float,
    y: float,
    label: str,
    *,
    color: str = "white",
) -> None:
    if right - left < 7.0:
        return
    ax.text(
        (left + right) / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=TREE_FONT_SIZE,
        fontweight="bold",
        color=color,
        zorder=7,
    )


def _tree_edge_annotation(
    ax: plt.Axes, x: float, label: str, *, above: bool
) -> None:
    if above:
        start = TARGET_Y + BAR_HEIGHT / 2
        ax.plot(
            [x, x], [start, start + 0.032],
            color=COLORS["text"], linewidth=3.6, zorder=6,
        )
        ax.text(
            x, start + 0.043, label,
            ha="center", va="bottom", fontsize=TREE_FONT_SIZE,
            fontweight="bold",
            color=COLORS["text"], zorder=6,
        )
    else:
        start = DRAFT_Y - BAR_HEIGHT / 2
        ax.plot(
            [x, x], [start - 0.042, start],
            color=COLORS["text"], linewidth=3.6, zorder=6,
        )
        ax.text(
            x, start - 0.050, label,
            ha="center", va="top", fontsize=TREE_FONT_SIZE,
            fontweight="bold",
            color=COLORS["text"], zorder=6,
        )


def _draw_tree_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    case: dict[str, float],
    *,
    cache_hit: bool,
) -> None:
    sync_end = case["sync_end"]
    stall_end = case["stall_end"]
    pre_end = case["verify_pre_end"]
    proxy_arrival = case["proxy_arrival"]
    post_end = case["verify_post_end"]
    glue_end = case["glue_end"]
    p1_end = case["p1_end"]
    p2_end = min(case["p2_end"], 100.0)

    # Keep the arrival marker behind both GPU timelines so it cannot obscure
    # a Verify/Draft label.  Its heavier stroke remains visible in the gaps.
    _vertical_marker(ax, proxy_arrival, linewidth=4.0, zorder=2)

    _bar(ax, TARGET_Y, 0, sync_end, COLORS["sync"], hatch="//")
    if not cache_hit:
        _bar(
            ax,
            TARGET_Y,
            sync_end,
            stall_end,
            COLORS["sync"],
            edgecolor=COLORS["redraft_edge"],
            linewidth=2.0,
            hatch="//",
        )
    _bar(ax, TARGET_Y, stall_end, pre_end, COLORS["verify_pre"])
    _bar(ax, TARGET_Y, pre_end, post_end, COLORS["verify_post"])
    _bar(ax, TARGET_Y, post_end, 100.0, COLORS["sample"])

    if cache_hit:
        _bar(ax, DRAFT_Y, 0, glue_end, COLORS["context"])
    else:
        redraft_end = case["draft_response_end"]
        _bar(
            ax,
            DRAFT_Y,
            0,
            redraft_end,
            COLORS["redraft"],
            edgecolor=COLORS["redraft_edge"],
            linewidth=2.0,
            hatch="//",
        )
        _bar(ax, DRAFT_Y, redraft_end, glue_end, COLORS["context"])
    _bar(ax, DRAFT_Y, glue_end, p1_end, COLORS["draft"])
    _bar(ax, DRAFT_Y, proxy_arrival, p2_end, COLORS["proxy_draft"])

    _tree_inside(
        ax,
        stall_end,
        pre_end,
        TARGET_Y,
        f"Verify (0–56) {_rounded_width(stall_end, pre_end)}%",
    )
    _tree_inside(
        ax,
        pre_end,
        post_end,
        TARGET_Y,
        f"Verify (57–80) {_rounded_width(pre_end, post_end)}%",
    )
    _tree_inside(
        ax,
        glue_end,
        p1_end,
        DRAFT_Y,
        f"Draft-source {_rounded_width(glue_end, p1_end)}%",
    )
    _tree_inside(
        ax,
        proxy_arrival,
        p2_end,
        DRAFT_Y,
        f"Proxy-source {_rounded_width(proxy_arrival, p2_end)}%",
    )

    if not cache_hit:
        redraft_end = case["draft_response_end"]
        # The bold 29-pt label is wider than the measured 13% bar.  Keep the
        # font size consistent with every other latency label and place it in
        # the free space between GPU tracks with a leader to the exact bar.
        ax.annotate(
            f"Re-draft {_rounded_width(0, redraft_end)}%",
            xy=(9.0, DRAFT_Y + BAR_HEIGHT / 2),
            xytext=(9.0, 0.505),
            ha="center",
            va="center",
            fontsize=TREE_FONT_SIZE,
            fontweight="bold",
            color=COLORS["redraft_edge"],
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["redraft_edge"],
                "linewidth": 3.6,
                "shrinkA": 3,
                "shrinkB": 0,
            },
            zorder=9,
        )
        stall = ax.text(
            (sync_end + stall_end) / 2,
            TARGET_Y,
            f"Stall {_rounded_width(sync_end, stall_end)}%",
            ha="center",
            va="center",
            fontsize=TREE_FONT_SIZE,
            fontweight="bold",
            color=COLORS["redraft_edge"],
            zorder=8,
        )
        stall.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])

    proxy_y0, proxy_y1 = 0.4464, 0.5456
    ax.add_patch(
        Rectangle(
            (pre_end, proxy_y0),
            proxy_arrival - pre_end,
            proxy_y1 - proxy_y0,
            facecolor=COLORS["proxy"],
            edgecolor="none",
            zorder=4,
        )
    )
    proxy_pct = proxy_arrival - pre_end
    # Keep the label independent from the orange latency block: no leader
    # line or arrow.  The miss label sits farther right because its marker is
    # closer to the long Verify label.
    ax.text(
        101.0 if cache_hit else 103.0,
        (proxy_y0 + proxy_y1) / 2,
        f"Early-exit Proxy {proxy_pct:.1f}%",
        ha="right",
        va="center",
        fontsize=TREE_FONT_SIZE,
        fontweight="bold",
        color=COLORS["annotation"],
        zorder=6,
    )
    ax.text(
        proxy_arrival - 0.65,
        0.835,
        "Proxy arrives",
        ha="right",
        va="bottom",
        fontsize=TREE_FONT_SIZE,
        fontweight="bold",
        color=COLORS["annotation"],
        zorder=6,
    )

    if proxy_arrival > p1_end:
        arrow_y = 0.24
        ax.annotate(
            "",
            xy=(proxy_arrival, arrow_y),
            xytext=(p1_end, arrow_y),
            arrowprops={
                "arrowstyle": "<->",
                "color": "#8a7d5a",
                "linewidth": 4.0,
                "mutation_scale": 14,
            },
        )
        ax.text(
            (p1_end + proxy_arrival) / 2,
            arrow_y + 0.058,
            f"{_rounded_width(p1_end, proxy_arrival)}%",
            ha="center",
            va="bottom",
            fontsize=TREE_FONT_SIZE,
            fontweight="bold",
            color=COLORS["headroom"],
        )

    _tree_edge_annotation(
        ax,
        sync_end / 2,
        f"Sync {_rounded_width(0, sync_end)}%",
        above=True,
    )
    _tree_edge_annotation(
        ax,
        min((post_end + 100.0) / 2, 96.6),
        f"Sample {_rounded_width(post_end, 100.0)}%",
        above=True,
    )
    glue_start = 0.0 if cache_hit else case["draft_response_end"]
    _tree_edge_annotation(
        ax,
        (glue_start + glue_end) / 2,
        f"Context Align {_rounded_width(glue_start, glue_end)}%",
        above=False,
    )
    ax.set_xlabel(
        "Fraction of one decoding step (%)",
        fontsize=TREE_AXIS_FONT_SIZE,
        labelpad=4,
    )


def build_tree_figure(data: dict[str, Any]) -> plt.Figure:
    _set_paper_rcparams()
    fig = plt.figure(
        figsize=(
            TREE_CANVAS_SCALE * PAGE_WIDTH_PT / 72.0,
            TREE_CANVAS_SCALE * PAGE_HEIGHT_PT / 72.0,
        ),
        dpi=PNG_DPI,
        facecolor="white",
    )
    top = _setup_axis(
        fig, TOP_BOTTOM_PT,
        left_pt=TREE_AX_LEFT_PT, width_pt=TREE_AX_WIDTH_PT,
    )
    bottom = _setup_axis(
        fig, TREE_BOTTOM_BOTTOM_PT,
        left_pt=TREE_AX_LEFT_PT, width_pt=TREE_AX_WIDTH_PT,
    )
    for ax in (top, bottom):
        ax.tick_params(axis="both", labelsize=TREE_AXIS_FONT_SIZE)
        ax.tick_params(axis="y", length=0, pad=3)
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        # Keep the large Context Align annotation clear of the x-axis ticks
        # without moving any measured timeline geometry.
        ax.spines["bottom"].set_position(("outward", 17))
    _draw_tree_panel(fig, top, data["cases"]["hit"], cache_hit=True)
    _draw_tree_panel(fig, bottom, data["cases"]["miss"], cache_hit=False)
    fig.text(
        *_figure_xy(TREE_AX_CENTER_PT, TREE_TOP_TITLE_Y_PT),
        "(a) Cache hit",
        ha="center",
        va="baseline",
        fontsize=TREE_PANEL_TITLE_SIZE,
        color="black",
    )
    fig.text(
        *_figure_xy(TREE_AX_CENTER_PT, 20.0),
        "(b) Cache miss",
        ha="center",
        va="baseline",
        fontsize=TREE_PANEL_TITLE_SIZE,
        color="black",
    )
    return fig


def _edge_annotation(
    fig: plt.Figure,
    line_x_pt: float,
    line_y0_pt: float,
    line_y1_pt: float,
    text_x_pt: float,
    text_y_pt: float,
    label: str,
) -> None:
    _figure_line(
        fig,
        line_x_pt,
        line_y0_pt,
        line_x_pt,
        line_y1_pt,
        color=COLORS["text"],
    )
    _figure_text(fig, text_x_pt, text_y_pt, label, color=COLORS["text"])


def _context_annotation(
    fig: plt.Figure,
    line_x_pt: float,
    line_y0_pt: float,
    line_y1_pt: float,
    text_x_pt: float,
    text_y_pt: float,
    label: str,
) -> None:
    _figure_line(
        fig,
        line_x_pt,
        line_y0_pt,
        line_x_pt,
        line_y1_pt,
        color=COLORS["text"],
    )
    _figure_text(fig, text_x_pt, text_y_pt, label, color=COLORS["text"])


def _set_paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 22,
            "axes.unicode_minus": True,
            "hatch.linewidth": 1.0,
            "pdf.fonttype": 3,
            "ps.fonttype": 3,
        }
    )


def build_figure() -> plt.Figure:
    _set_paper_rcparams()
    fig = plt.figure(
        figsize=(PAGE_WIDTH_PT / 72.0, PAGE_HEIGHT_PT / 72.0),
        dpi=PNG_DPI,
        facecolor="white",
    )
    top = _setup_axis(fig, TOP_BOTTOM_PT)
    bottom = _setup_axis(fig, BOTTOM_BOTTOM_PT)
    _top_panel(fig, top)
    _bottom_panel(fig, bottom)
    return fig


def _write_tree_manifest(path: Path, data: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    definitions = {
        "hit": (
            "hit_k2",
            (
                ("Sync", 0.0, "sync_end"),
                ("Verify (0–56)", "stall_end", "verify_pre_end"),
                ("Early-exit Proxy", "verify_pre_end", "proxy_arrival"),
                ("Verify (57–80)", "verify_pre_end", "verify_post_end"),
                ("Sample", "verify_post_end", "sample_end"),
                ("Context Align", 0.0, "glue_end"),
                ("Draft-source", "glue_end", "p1_end"),
                ("Proxy-source", "proxy_arrival", "p2_end"),
            ),
        ),
        "miss": (
            "miss",
            (
                ("Sync", 0.0, "sync_end"),
                ("Stall", "sync_end", "stall_end"),
                ("Verify (0–56)", "stall_end", "verify_pre_end"),
                ("Early-exit Proxy", "verify_pre_end", "proxy_arrival"),
                ("Verify (57–80)", "verify_pre_end", "verify_post_end"),
                ("Sample", "verify_post_end", "sample_end"),
                ("Re-draft", 0.0, "draft_response_end"),
                ("Context Align", "draft_response_end", "glue_end"),
                ("Draft-source", "glue_end", "p1_end"),
                ("Proxy-source", "proxy_arrival", "p2_end"),
            ),
        ),
    }
    for panel, (status, components) in definitions.items():
        case = data["cases"][panel]
        total_ms = case["total_ms"]
        for component, start_key, end_key in components:
            start_pct = float(start_key) if isinstance(start_key, float) else case[start_key]
            end_pct = min(case[end_key], 100.0)
            rows.append(
                {
                    "panel": panel,
                    "source_status": status,
                    "component": component,
                    "start_pct": f"{start_pct:.6f}",
                    "end_pct": f"{end_pct:.6f}",
                    "duration_ms": f"{(end_pct - start_pct) * total_ms / 100:.6f}",
                    "total_step_ms": f"{total_ms:.6f}",
                    "common_target_verify_ms": f"{data['verify_ms']:.6f}",
                    "target_verify_bucket_nodes": data["verify_nodes"],
                    "target_verify_rows": data["verify_rows"],
                    "samples": int(case["sample_count"]),
                    "profile_dir": str(data["profile_dir"]),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_dir = Path(__file__).resolve().parents[2] / "tmp/duet_tree_timeline/reproduced"
    parser.add_argument(
        "--variant",
        choices=("chain", "tree"),
        default="chain",
        help="reproduce the original chain figure or build the matched-width tree figure",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--stem", default=None)
    parser.add_argument(
        "--tree-profile-dir",
        type=Path,
        default=DEFAULT_TREE_PROFILE_DIR,
        help="aligned P1+P2 tree profile used by --variant tree",
    )
    parser.add_argument(
        "--skip-request-epochs",
        type=int,
        default=2,
        help="warmup generate() calls excluded from tree category means",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == "tree":
        data = load_tree_figure_data(
            args.tree_profile_dir, args.skip_request_epochs
        )
        fig = build_tree_figure(data)
        stem = args.stem or "paper_fig4_tree_schematic_pct"
        manifest = args.output_dir / f"{stem}.csv"
        _write_tree_manifest(manifest, data)
    else:
        data = None
        fig = build_figure()
        stem = args.stem or "paper_fig4_schematic_pct"
        manifest = None
    png = args.output_dir / f"{stem}.png"
    pdf = args.output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)
    if manifest is not None:
        print(manifest)
        print(
            "matched target verification: "
            f"P2 hit M2={data['verify_nodes']} == miss K1={data['verify_nodes']} "
            f"({data['verify_rows']} rows including recovery); "
            f"common verify={data['verify_ms']:.3f} ms; "
            f"target Stall={data['stall_ms']:.3f} ms; "
            f"draft Re-draft={data['redraft_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
