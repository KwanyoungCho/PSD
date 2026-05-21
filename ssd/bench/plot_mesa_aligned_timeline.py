"""Aligned MESA timeline plotter (Phase C).

Consumes the Phase-B JSON schema (rows with ``_anchor`` sentinel,
``wall_start_ns`` / ``wall_end_ns``, ``step_id``, ``parent_label``, ``status``)
and produces a paper-quality two-row (target / draft) Gantt for a single
speculative step, joined by ``step_id``.

Importable from ``summarize_ssd_run.py`` via ``plot_aligned_for_status``,
also runnable as::

    python bench/plot_mesa_aligned_timeline.py OUTDIR --step-id N --out path.png

Rules (see ``ssd/docs/mesa/06-timeline-cleanup-plan.md`` §4.6):

- Select target rank0 by ``step_id`` and include draft spans that overlap the
  target request→response window.  Cache-hit waits often overlap the previous
  draft step's phase2 replay, so strict same-step draft filtering is
  misleading.
- X-axis is ms-from-step-origin computed from ``wall_start_ns``.
- Render parents translucent; inner spans (e.g. ``target_spec_wait`` →
  send/recv markers, ``phase2_build`` → ``proxy_wait``) drawn on top.
- Draw a faint arrow from ``draft_send_response.end`` to
  ``target_response_received.start`` showing the causal link.
- One PNG per cache status: ``timeline_cache_hit_k1.png``,
  ``timeline_cache_hit_k2.png``, ``timeline_cache_miss.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Extends the COLORS dict in summarize_ssd_run.py.  Same hex values are reused
# where the label already existed there; the new entries cover the Phase-B
# handshake markers.
COLORS: dict[str, tuple[str, str]] = {
    # Target-side MESA events
    "verify_setup": ("#969696", ".."),
    "graph_pre": ("#8b1a1a", ""),
    "exit_logits": ("#fdae61", ""),
    "proxy_compute_send": ("#1b9e77", ""),
    "proxy_compute": ("#66c2a5", ""),
    "proxy_pack": ("#1b9e77", ".."),
    "proxy_send": ("#006d2c", ""),
    "graph_post": ("#8073ac", ""),
    "final_logits": ("#fee08b", ""),
    "verify_replay": ("#7a0000", ""),
    "verify_sample_accept": ("#fc8d59", ""),
    "target_postprocess": ("#feb24c", ".."),
    # Target spec wait parents
    "target_spec_wait": ("#ffeda0", ""),
    "target_spec_wait_hit": ("#c7e9c0", ""),
    "target_spec_wait_hit_k1": ("#74c476", ""),
    "target_spec_wait_hit_k2": ("#238b45", ""),
    "target_spec_wait_miss": ("#fdd0a2", ""),
    "target_spec_wait_mixed": ("#d9d9d9", "xx"),
    # Phase-B handshake markers (inside target_spec_wait)
    "target_send_request": ("#3182bd", ""),
    "target_recv_response_wait": ("#fdae6b", ""),
    "target_response_received": ("#08519c", ""),
    # Draft-side
    "tree_prep": ("#feb24c", "//"),
    "tree_replay": ("#e6550d", ""),
    "glue": ("#01665e", ""),
    "draft_glue_replay": ("#5ab4ac", ""),
    "hit_cache_respond": ("#80cdc1", "//"),
    "hit_cache_respond_hit": ("#80cdc1", "//"),
    "hit_cache_respond_hit_k1": ("#67a9cf", "//"),
    "hit_cache_respond_hit_k2": ("#2166ac", "//"),
    "hit_cache_respond_miss": ("#fb6a4a", "//"),
    "hit_cache_respond_mixed": ("#bdbdbd", "//"),
    "draft_recv_cmd": ("#4d4d4d", "..."),
    "draft_recv_request": ("#737373", "..."),
    "draft_send_response": ("#252525", ""),
    "phase1_build": ("#9ecae1", ".."),
    "phase1_prep": ("#6baed6", "//"),
    "phase1_replay": ("#08519c", ""),
    "proxy_wait": ("#ffd92f", ""),
    "phase2_build": ("#a1d99b", ".."),
    "phase2_prep": ("#74c476", "//"),
    "phase2_replay": ("#006d2c", ""),
    "merge_cache": ("#bcbddc", "\\\\"),
}


_DEFAULT_COLOR = ("#d9d9d9", "")


def _style(label: str) -> tuple[str, str]:
    return COLORS.get(label, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Schema detection / loading
# ---------------------------------------------------------------------------


class AlignedSchemaError(Exception):
    """Raised when a JSON file does not satisfy the Phase-B aligned schema."""


def _latest_json(outdir: Path, tag: str) -> Path | None:
    paths = sorted(outdir.glob(f"mesa_profile_{tag}_*.json"))
    if not paths:
        paths = sorted(outdir.glob(f"mesa_profile_{tag}.json"))
    return paths[-1] if paths else None


def _load_json(outdir: Path, tag: str) -> list[dict]:
    path = _latest_json(outdir, tag)
    if path is None:
        raise FileNotFoundError(f"no mesa_profile_{tag}_*.json in {outdir}")
    with path.open() as f:
        return json.load(f)


def is_aligned_schema(rows: list[dict]) -> bool:
    """Return True iff rows look like Phase-B aligned schema.

    Either an ``_anchor`` sentinel exists, OR every non-anchor row carries
    both ``wall_start_ns`` and ``step_id``.  We deliberately accept the
    weaker condition because the anchor row is best-effort metadata while
    the wall/step fields are the load-bearing alignment surface.
    """
    if not rows:
        return False
    has_anchor = any(r.get("label") == "_anchor" for r in rows)
    if has_anchor:
        return True
    non_anchor = [r for r in rows if r.get("label") != "_anchor"]
    if not non_anchor:
        return False
    return all(
        r.get("wall_start_ns") is not None and r.get("step_id") is not None
        for r in non_anchor
    )


def _strip_anchor(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("label") != "_anchor"]


def _anchor_row(rows: list[dict]) -> dict | None:
    for r in rows:
        if r.get("label") == "_anchor":
            return r
    return None


# ---------------------------------------------------------------------------
# Step selection
# ---------------------------------------------------------------------------


def select_step(
    target_rows: list[dict],
    draft_rows: list[dict],
    step_id: int,
    status_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return target rows for ``step_id`` and draft rows relevant to it.

    Target spans are keyed by the request step.  Draft spans are not always
    keyed the same way on the critical path: on cache-hit steps the target can
    wait for the draft to finish the previous step's phase2 replay before the
    current request is handled.  Therefore the draft subset includes rows that
    overlap the target wait window, not only rows with the identical
    ``step_id``.

    If ``status_filter`` is given, the *step* must match.  We use the
    ``target_spec_wait_*`` row's status (or, if absent, any row's status)
    as the step's canonical status — every span produced inside the same
    step inherits the same context (see Phase-B RESULTS step 50 example).
    """
    tgt = [r for r in target_rows if r.get("step_id") == step_id]
    drf = [r for r in draft_rows if r.get("step_id") == step_id]
    if status_filter is not None:
        # Determine the step's canonical status from the target spec_wait row.
        step_status = None
        for r in tgt:
            lbl = str(r.get("label", ""))
            if lbl.startswith("target_spec_wait"):
                step_status = r.get("status")
                break
        if step_status is None:
            for r in tgt + drf:
                if r.get("status"):
                    step_status = r.get("status")
                    break
        if step_status != status_filter:
            return [], []

    win = _target_wait_window_ns(tgt)
    if win is not None:
        drf_overlap = _rows_overlapping_window(draft_rows, win)
        if drf_overlap:
            # Keep deterministic order and avoid duplicates when same-step rows
            # also overlap the window.
            seen: set[int] = set()
            merged: list[dict] = []
            for r in drf + drf_overlap:
                rid = id(r)
                if rid in seen:
                    continue
                seen.add(rid)
                merged.append(r)
            drf = sorted(
                merged,
                key=lambda r: (
                    int(r.get("wall_start_ns") or 0),
                    int(r.get("idx") or 0),
                ),
            )
    return tgt, drf


def _target_wait_window_ns(target_rows: list[dict]) -> tuple[int, int] | None:
    """Return the target request→response window for one selected step.

    Prefer explicit handshake markers.  Fall back to the parent
    ``target_spec_wait*`` span so older aligned traces still produce a useful
    plot.
    """
    send = next(
        (r for r in target_rows if str(r.get("label", "")) == "target_send_request"),
        None,
    )
    recv = next(
        (
            r
            for r in target_rows
            if str(r.get("label", "")) == "target_response_received"
        ),
        None,
    )
    if send is not None and recv is not None:
        s = send.get("wall_start_ns")
        e = recv.get("wall_start_ns")
        if s is not None and e is not None and int(e) >= int(s):
            return int(s), int(e)

    parent = next(
        (
            r
            for r in target_rows
            if str(r.get("label", "")).startswith("target_spec_wait")
        ),
        None,
    )
    if parent is None:
        return None
    s = parent.get("wall_start_ns")
    e = parent.get("wall_end_ns")
    if s is None or e is None or int(e) < int(s):
        return None
    return int(s), int(e)


def _rows_overlapping_window(
    rows: list[dict],
    window_ns: tuple[int, int],
    *,
    margin_ns: int = 0,
) -> list[dict]:
    """Rows whose wall-clock interval intersects ``window_ns``."""
    lo, hi = window_ns
    lo -= margin_ns
    hi += margin_ns
    out: list[dict] = []
    for r in rows:
        s = r.get("wall_start_ns")
        e = r.get("wall_end_ns")
        if s is None or e is None:
            continue
        if min(int(e), hi) > max(int(s), lo):
            out.append(r)
    return out


def step_full_duration_ms(rows: list[dict]) -> float | None:
    """Span across all rows of a single step using wall_start_ns / wall_end_ns."""
    walls = [
        (r.get("wall_start_ns"), r.get("wall_end_ns"))
        for r in rows
        if r.get("wall_start_ns") is not None and r.get("wall_end_ns") is not None
    ]
    if not walls:
        return None
    lo = min(s for s, _ in walls)
    hi = max(e for _, e in walls)
    return (hi - lo) / 1e6


def list_steps_by_status(
    target_rows: list[dict],
) -> dict[str, list[tuple[int, float]]]:
    """Return {status: [(step_id, full_step_ms)]} sorted by full_step_ms."""
    # canonical status per step from target_spec_wait_* row
    status_by_step: dict[int, str | None] = {}
    rows_by_step: dict[int, list[dict]] = {}
    for r in target_rows:
        sid = r.get("step_id")
        if sid is None:
            continue
        rows_by_step.setdefault(sid, []).append(r)
        lbl = str(r.get("label", ""))
        if lbl.startswith("target_spec_wait"):
            status_by_step[sid] = r.get("status")

    out: dict[str, list[tuple[int, float]]] = {}
    for sid, status in status_by_step.items():
        if status is None:
            continue
        full = step_full_duration_ms(rows_by_step[sid])
        if full is None:
            continue
        out.setdefault(status, []).append((sid, full))
    for status in out:
        out[status].sort(key=lambda t: t[1])
    return out


def pick_representative_step(
    target_rows: list[dict], status: str
) -> int | None:
    """Pick the median-full_step_ms step for ``status``."""
    table = list_steps_by_status(target_rows)
    items = table.get(status)
    if not items:
        return None
    return items[len(items) // 2][0]


# ---------------------------------------------------------------------------
# Causality-based draft shift (corrects CUDA clock drift over long runs)
# ---------------------------------------------------------------------------
#
# Each process anchors its own CUDA event timer at startup.  Over a long run
# the GPU clocks of target rank 0 and draft drift apart (~1-10 ppm typical),
# accumulating ms-level skew that breaks cross-process wall_ns alignment in
# the trace.  Symptom: draft_send_response.wall_end_ns > target_response_
# received.wall_start_ns for the same step_id (target appears to receive
# before draft sent), which is physically impossible.
#
# For paper-figure renders we correct this *visually* by shifting the draft
# row by the median offset measured from response causal pairs around the
# selected step.  JSON data is untouched.  The title carries a "causality-
# shifted by X ms" annotation so readers know.
#
# Why the response pair (draft_send_response.end → target_response_received
# .start) is the right anchor: it is the only event pair that REQUIRES strict
# wall-time ordering on both sides (data must be physically sent before being
# received).  target_send_request → draft_recv_request is a weaker constraint
# because the draft side may have posted recv long before the send fired.


def _find_step_event(
    rows: list[dict], step_id: int, label: str
) -> dict | None:
    for r in rows:
        if r.get("step_id") == step_id and str(r.get("label", "")) == label:
            return r
    return None


def _step_response_pair_offset_ns(
    target_rows: list[dict], draft_rows: list[dict], step_id: int
) -> int | None:
    """One step's causality offset = draft_send_response.end − target_response_received.start.

    Positive value → causality violation (draft appears to send AFTER target
    received); the draft row needs to be shifted *backward* by this amount.
    Returns ``None`` if either marker is missing.
    """
    send = _find_step_event(draft_rows, step_id, "draft_send_response")
    recv = _find_step_event(target_rows, step_id, "target_response_received")
    if send is None or recv is None:
        return None
    we = send.get("wall_end_ns")
    rs = recv.get("wall_start_ns")
    if we is None or rs is None:
        return None
    return int(we) - int(rs)


def compute_causality_shift_ns(
    target_rows: list[dict],
    draft_rows: list[dict],
    center_step_id: int,
    window: int = 5,
) -> tuple[int, int]:
    """Median draft-side shift (in ns) over a window of response pairs.

    Looks at step_ids in [center − window, center + window] (inclusive,
    skipping those without a complete response pair).  Returns
    ``(shift_ns, n_pairs_used)``.  Positive ``shift_ns`` means draft events
    should be displayed ``shift_ns`` *earlier* (subtract from wall_start/end).

    A single-pair estimate is noisy; the window median rejects outliers from
    network jitter and aligns with the ambient drift at this point in the run.
    """
    candidates: list[int] = []
    for sid in range(center_step_id - window, center_step_id + window + 1):
        off = _step_response_pair_offset_ns(target_rows, draft_rows, sid)
        if off is not None:
            candidates.append(off)
    if not candidates:
        return 0, 0
    candidates.sort()
    n = len(candidates)
    median = candidates[n // 2] if n % 2 else (candidates[n // 2 - 1] + candidates[n // 2]) // 2
    return int(median), n


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


_HANDSHAKE_CHILDREN = {
    "target_send_request",
    "target_recv_response_wait",
    "target_response_received",
}


def _origin_ns(target_rows: list[dict], draft_rows: list[dict]) -> int:
    starts = [
        int(r["wall_start_ns"])
        for r in target_rows + draft_rows
        if r.get("wall_start_ns") is not None
    ]
    if not starts:
        return 0
    return min(starts)


def _ms_from_origin(ns: int | float | None, origin: int) -> float | None:
    if ns is None:
        return None
    return (int(ns) - origin) / 1e6


def _draw_row(
    ax,
    rows: list[dict],
    y_parent: float,
    y_child: float,
    origin: int,
    bar_h_parent: float,
    bar_h_child: float,
    shift_ns: int = 0,
) -> None:
    """Draw one process row.  Parents go on bottom (translucent), children on top.

    ``shift_ns`` is subtracted from each row's wall_start_ns before plotting —
    used to apply the causality correction to the draft row without touching
    the JSON data.  Positive shift_ns moves bars to the LEFT.
    """
    # Identify parent set.  In the Phase-B schema only two parents exist:
    #   target_spec_wait_*  and  phase2_build.
    # Children carry parent_label="target_spec_wait" or "phase2_build".
    parent_labels = {"target_spec_wait", "phase2_build"}
    for r in rows:
        lbl = str(r.get("label", ""))
        ws = r.get("wall_start_ns")
        we = r.get("wall_end_ns")
        if ws is None or we is None:
            continue
        x0 = (int(ws) - shift_ns - origin) / 1e6
        # Prefer cuda_ms for the bar width (CUDA-event duration is the
        # authoritative per-event GPU duration; wall_end - wall_start should
        # match closely but using cuda_ms keeps duration labels consistent
        # with summary tables).
        cuda_ms = r.get("cuda_ms")
        if cuda_ms is None:
            cuda_ms = (int(we) - int(ws)) / 1e6
        w = max(float(cuda_ms), 0.02)
        color, hatch = _style(lbl)

        parent_label = r.get("parent_label")
        is_parent = (
            parent_label is None
            and (lbl.startswith("target_spec_wait") or lbl == "phase2_build")
        )

        if is_parent:
            ax.barh(
                y_parent,
                w,
                left=x0,
                height=bar_h_parent,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.35,
                zorder=1,
            )
        elif parent_label in parent_labels:
            # Inner span — draw on top
            ax.barh(
                y_child,
                w,
                left=x0,
                height=bar_h_child,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.95,
                zorder=3,
            )
        else:
            # Regular root span (graph_pre, exit_logits, ..., phase1_replay, ...)
            ax.barh(
                y_parent,
                w,
                left=x0,
                height=bar_h_parent,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
                zorder=2,
            )


def _draw_causality_arrow(
    ax,
    target_rows: list[dict],
    draft_rows: list[dict],
    y_target: float,
    y_draft: float,
    origin: int,
    draft_shift_ns: int = 0,
) -> None:
    """Faint arrow: draft_send_response.end --> target_response_received.start.

    The draft endpoint is shifted by ``draft_shift_ns`` to match the visual
    correction applied to the draft row.
    """
    send = next(
        (r for r in draft_rows if str(r.get("label", "")) == "draft_send_response"),
        None,
    )
    recv = next(
        (
            r
            for r in target_rows
            if str(r.get("label", "")) == "target_response_received"
        ),
        None,
    )
    if send is None or recv is None:
        return
    we = send.get("wall_end_ns")
    rs = recv.get("wall_start_ns")
    if we is None or rs is None:
        return
    x_send = (int(we) - draft_shift_ns - origin) / 1e6
    x_recv = (int(rs) - origin) / 1e6
    ax.annotate(
        "",
        xy=(x_recv, y_target),
        xytext=(x_send, y_draft),
        arrowprops=dict(
            arrowstyle="->",
            color="#444",
            alpha=0.55,
            lw=1.0,
            connectionstyle="arc3,rad=-0.15",
        ),
        zorder=4,
    )


def plot_aligned_step(
    target_rows: list[dict],
    draft_rows: list[dict],
    step_id: int,
    out_path: Path,
    title_suffix: str = "",
    draft_shift_ns: int = 0,
    shift_n_pairs: int = 0,
) -> Path:
    """Render the two-row aligned timeline for one step_id.

    ``target_rows`` / ``draft_rows`` are the per-step subsets returned by
    ``select_step``.  Saves PNG to ``out_path`` and returns the path.

    Args:
        draft_shift_ns: visual correction applied to draft row (subtracted
            from each draft event's wall_start_ns).  Used to compensate for
            CUDA-event clock drift between target and draft GPUs on long
            runs.  See ``compute_causality_shift_ns``.  JSON data is NOT
            modified.  When non-zero, the title annotates the shift.
        shift_n_pairs: number of response pairs used to compute the shift
            (for the title annotation).  0 means no shift information.
    """
    if not target_rows and not draft_rows:
        raise ValueError(f"no rows for step_id={step_id}")

    origin = _origin_ns(target_rows, draft_rows)

    fig, ax = plt.subplots(figsize=(16, 4.6))
    y_target_parent = 1.0
    y_target_child = 1.0  # children sit on top of parent at same y (zorder split)
    y_draft_parent = 0.0
    y_draft_child = 0.0

    _draw_row(
        ax,
        target_rows,
        y_target_parent,
        y_target_child,
        origin,
        bar_h_parent=0.65,
        bar_h_child=0.45,
        shift_ns=0,  # target is the reference, never shifted
    )
    _draw_row(
        ax,
        draft_rows,
        y_draft_parent,
        y_draft_child,
        origin,
        bar_h_parent=0.65,
        bar_h_child=0.45,
        shift_ns=draft_shift_ns,
    )

    _draw_causality_arrow(
        ax, target_rows, draft_rows, y_target_parent, y_draft_parent, origin,
        draft_shift_ns=draft_shift_ns,
    )

    # X extent: account for the draft shift too so bars are not clipped
    all_ends = []
    for r in target_rows:
        we = r.get("wall_end_ns")
        if we is not None:
            all_ends.append((int(we) - origin) / 1e6)
    for r in draft_rows:
        we = r.get("wall_end_ns")
        if we is not None:
            all_ends.append((int(we) - draft_shift_ns - origin) / 1e6)
    xmax = max(all_ends) if all_ends else 1.0

    # X-min: shifted draft bars may extend to negative x
    all_starts = [0.0]
    for r in draft_rows:
        ws = r.get("wall_start_ns")
        if ws is not None:
            all_starts.append((int(ws) - draft_shift_ns - origin) / 1e6)
    xmin = min(all_starts)

    ax.set_yticks([y_draft_parent, y_target_parent])
    ax.set_yticklabels(["draft", "target_rank0"])
    ax.set_xlabel("time (ms from step start, host monotonic clock)")
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(-0.7, 1.7)
    title = f"MESA aligned timeline — step_id={step_id}"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    if draft_shift_ns != 0 and shift_n_pairs > 0:
        title = (
            f"{title}  [draft causality-shifted {draft_shift_ns/1e6:+.3f} ms "
            f"(median over {shift_n_pairs} response pairs)]"
        )
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Legend — keep only labels actually drawn.
    present_labels = sorted(
        {str(r.get("label", "")) for r in target_rows + draft_rows}
    )
    handles = []
    for label in present_labels:
        color, hatch = _style(label)
        handles.append(
            mpatches.Patch(
                facecolor=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                label=label,
                alpha=0.9,
            )
        )
    if handles:
        n_cols = min(6, max(3, (len(handles) + 1) // 2))
        ax.legend(
            handles=handles,
            fontsize=7.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=n_cols,
            frameon=True,
            framealpha=0.9,
        )
    plt.subplots_adjust(bottom=0.32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public dispatch helpers (used by summarize_ssd_run.py)
# ---------------------------------------------------------------------------


_STATUS_SUFFIX = {
    "hit_k1": "timeline_cache_hit_k1.png",
    "hit_k2": "timeline_cache_hit_k2.png",
    "miss": "timeline_cache_miss.png",
}


def plot_aligned_for_status(
    outdir: Path,
    target_rows: list[dict],
    draft_rows: list[dict],
    status: str,
    *,
    causality_shift: bool = False,
    causality_window: int = 5,
) -> Path | None:
    """Generate ``timeline_cache_<status>.png`` for one status.

    Returns the written path, or ``None`` if no step with that status exists.

    Args:
        causality_shift: if True, shift the draft row by the median
            ``draft_send_response.end − target_response_received.start``
            offset over ±``causality_window`` steps around the rendered step.
            Corrects CUDA-event clock drift between target and draft GPUs
            on long runs.  See ``compute_causality_shift_ns``.
    """
    step_id = pick_representative_step(target_rows, status)
    if step_id is None:
        return None
    tgt, drf = select_step(target_rows, draft_rows, step_id, status_filter=status)
    if not tgt:
        return None
    shift_ns = 0
    n_pairs = 0
    if causality_shift:
        shift_ns, n_pairs = compute_causality_shift_ns(
            target_rows, draft_rows, step_id, window=causality_window,
        )
    fname = _STATUS_SUFFIX.get(status, f"timeline_cache_{status}.png")
    out_path = outdir / fname
    return plot_aligned_step(
        tgt, drf, step_id, out_path,
        title_suffix=f"cache {status.replace('_', ' ')}",
        draft_shift_ns=shift_ns,
        shift_n_pairs=n_pairs,
    )


def plot_all_statuses(
    outdir: Path,
    *,
    causality_shift: bool = False,
    causality_window: int = 5,
) -> dict[str, Path | None]:
    """Top-level helper used by ``summarize_ssd_run.py`` when aligned schema is detected.

    Returns ``{status: path-or-None}`` for hit_k1, hit_k2, miss.
    Raises ``AlignedSchemaError`` if the target JSON is not aligned schema.
    """
    target_rows = _load_json(outdir, "target_rank0")
    draft_rows = _load_json(outdir, "draft")
    if not is_aligned_schema(target_rows):
        raise AlignedSchemaError(
            f"target JSON in {outdir} is not Phase-B aligned schema "
            "(_anchor row + wall_start_ns + step_id required)"
        )
    target_rows = _strip_anchor(target_rows)
    draft_rows = _strip_anchor(draft_rows)
    out: dict[str, Path | None] = {}
    for status in ("hit_k1", "hit_k2", "miss"):
        path = plot_aligned_for_status(
            outdir, target_rows, draft_rows, status,
            causality_shift=causality_shift,
            causality_window=causality_window,
        )
        out[status] = path
    return out


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    k = q * (len(vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def print_status_summary(target_rows: list[dict]) -> None:
    """Per-status counts + target_spec_wait mean/median/p99 (cuda_ms)."""
    by_status_wait: dict[str, list[float]] = {}
    step_status: dict[int, str] = {}
    for r in target_rows:
        lbl = str(r.get("label", ""))
        if not lbl.startswith("target_spec_wait"):
            continue
        s = r.get("status")
        if s is None:
            continue
        ms = float(r.get("cuda_ms", r.get("ms", 0.0)))
        by_status_wait.setdefault(s, []).append(ms)
        sid = r.get("step_id")
        if sid is not None:
            step_status[sid] = s

    counts: dict[str, int] = {}
    for s in step_status.values():
        counts[s] = counts.get(s, 0) + 1

    print("by_status summary (target rank0):")
    print(f"  {'status':<10} {'n_steps':>8} {'wait_mean':>10} {'wait_median':>12} {'wait_p99':>10}")
    for s in sorted(by_status_wait):
        vals = by_status_wait[s]
        n = counts.get(s, len(vals))
        print(
            f"  {s:<10} {n:>8d} "
            f"{statistics.fmean(vals):>10.3f} "
            f"{statistics.median(vals):>12.3f} "
            f"{_percentile(vals, 0.99):>10.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path, help="directory containing mesa_profile_*.json")
    ap.add_argument(
        "--step-id",
        type=int,
        default=None,
        help="render only this step_id (default: render representative steps for each status)",
    )
    ap.add_argument(
        "--status",
        choices=("hit_k1", "hit_k2", "miss"),
        default=None,
        help="when --step-id is set, override the title's status string",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG path (used only with --step-id; otherwise three PNGs go into OUTDIR)",
    )
    ap.add_argument(
        "--causality-shift",
        action="store_true",
        help=(
            "visually shift the draft row by the median "
            "(draft_send_response.end − target_response_received.start) "
            "offset over ±window steps around the rendered step. Corrects "
            "CUDA-event clock drift between target and draft GPUs on long "
            "runs.  JSON data is NOT modified — only the plot."
        ),
    )
    ap.add_argument(
        "--causality-window",
        type=int,
        default=5,
        help="±N step window for median offset (default 5)",
    )
    args = ap.parse_args()

    target_rows = _load_json(args.outdir, "target_rank0")
    draft_rows = _load_json(args.outdir, "draft")

    if not is_aligned_schema(target_rows):
        print(
            f"ERROR: target JSON in {args.outdir} is not Phase-B aligned schema "
            "(missing _anchor / wall_start_ns / step_id).",
            file=sys.stderr,
        )
        sys.exit(2)

    target_rows = _strip_anchor(target_rows)
    draft_rows = _strip_anchor(draft_rows)

    print_status_summary(target_rows)

    if args.step_id is not None:
        tgt, drf = select_step(target_rows, draft_rows, args.step_id)
        if not tgt:
            print(
                f"ERROR: no rows for step_id={args.step_id} in target JSON",
                file=sys.stderr,
            )
            sys.exit(3)
        # Infer status from the step
        step_status = args.status
        if step_status is None:
            for r in tgt:
                if str(r.get("label", "")).startswith("target_spec_wait"):
                    step_status = r.get("status") or "unknown"
                    break
        out_path = args.out or (
            args.outdir / f"timeline_step{args.step_id}.png"
        )
        shift_ns = 0
        n_pairs = 0
        if args.causality_shift:
            shift_ns, n_pairs = compute_causality_shift_ns(
                target_rows, draft_rows, args.step_id,
                window=args.causality_window,
            )
            print(f"[causality-shift] draft shift = {shift_ns/1e6:+.3f} ms "
                  f"(median over {n_pairs} response pairs)")
        path = plot_aligned_step(
            tgt,
            drf,
            args.step_id,
            out_path,
            title_suffix=f"cache {step_status}" if step_status else "",
            draft_shift_ns=shift_ns,
            shift_n_pairs=n_pairs,
        )
        print(f"-> saved {path}")
        return

    # No specific step requested → render the three status PNGs
    for status in ("hit_k1", "hit_k2", "miss"):
        path = plot_aligned_for_status(
            args.outdir, target_rows, draft_rows, status,
            causality_shift=args.causality_shift,
            causality_window=args.causality_window,
        )
        if path is None:
            print(f"[skip] no step with status={status}")
        else:
            print(f"-> saved {path}")


if __name__ == "__main__":
    main()
