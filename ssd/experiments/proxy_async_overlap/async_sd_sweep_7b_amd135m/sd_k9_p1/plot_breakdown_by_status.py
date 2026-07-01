#!/usr/bin/env python3
"""Plot MESA target+draft per-event breakdown by status (hit_k1/hit_k2/miss).

Reads mesa_profile_target_rank0_*.json + mesa_profile_draft_*.json from OUTDIR
and emits:
  - target_breakdown_by_status.csv / .png
  - draft_breakdown_by_status.csv  / .png
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(sys.argv[1])

# Target events in pipeline order. MESA has graph_pre + exit_logits +
# proxy_compute_send + graph_post; vanilla async SD has only verify_replay.
TARGET_EVENTS = [
    "verify_setup",
    "graph_pre",            # MESA
    "exit_logits",          # MESA
    "proxy_compute_send",   # MESA (combined when SSD_PROFILE_MESA_DETAIL=0)
    "graph_post",           # MESA
    "verify_replay",        # async SD (monolithic forward)
    "final_logits",
    "verify_sample_accept",
    "target_postprocess",
]

# Draft events. MESA splits into phase1 + phase2 + proxy_wait; SD uses
# tree_prep/tree_replay summed across K per-step invocations.
DRAFT_EVENTS = [
    "draft_recv_cmd",
    "glue",
    "draft_glue_replay",
    "phase1_build",
    "phase1_prep",
    "phase1_replay",
    "proxy_wait",
    "phase2_build",
    "phase2_prep",
    "phase2_replay",
    "tree_prep",            # async SD
    "tree_replay",          # async SD
    "merge_cache",
    "hit_cache_respond",
    "draft_send_response",
]

# Colors per event group (target).
TARGET_COLORS = {
    "verify_setup": "#9ca3af",
    "graph_pre": "#3b82f6",
    "exit_logits": "#fbbf24",
    "proxy_compute_send": "#f97316",
    "graph_post": "#1d4ed8",
    "verify_replay": "#2563eb",
    "final_logits": "#d1d5db",
    "verify_sample_accept": "#10b981",
    "target_postprocess": "#6b7280",
}

DRAFT_COLORS = {
    "draft_recv_cmd": "#9ca3af",
    "glue": "#c084fc",
    "draft_glue_replay": "#a855f7",
    "phase1_build": "#fde68a",
    "phase1_prep": "#fcd34d",
    "phase1_replay": "#3b82f6",
    "proxy_wait": "#ef4444",
    "phase2_build": "#fef3c7",
    "phase2_prep": "#fbbf24",
    "phase2_replay": "#1d4ed8",
    "tree_prep": "#fbbf24",
    "tree_replay": "#3b82f6",
    "merge_cache": "#a3e635",
    "hit_cache_respond": "#10b981",
    "draft_send_response": "#6b7280",
}


def load_rows(pattern: str):
    matches = sorted(OUTDIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {pattern} in {OUTDIR}")
    with open(matches[-1]) as f:
        rows = json.load(f)
    return [r for r in rows if r.get("label") != "_anchor"]


def _get_dur(r):
    """Prefer cuda_ms, fall back to ms (older instrumentation)."""
    v = r.get("cuda_ms")
    if v is None:
        v = r.get("ms", 0)
    return float(v)


def _assign_sequential_step_ids(rows, status_marker_prefix):
    """For older JSONs without step_id, assign step_id by scanning event order.

    A new step starts each time we see a label with `status_marker_prefix`
    (e.g. "target_spec_wait_" or "hit_cache_respond_"). All events sharing
    the same step get the same synthetic step_id, and the marker's suffix
    becomes the step's status.
    Modifies rows in place.
    """
    status_of = {}
    cur_sid = -1
    cur_status = None
    # First pass: walk rows in idx order if present, else as-is.
    rows_sorted = sorted(rows, key=lambda r: r.get("idx", 0)) if "idx" in rows[0] else rows
    for r in rows_sorted:
        lbl = r.get("label", "")
        if lbl.startswith(status_marker_prefix):
            cur_sid += 1
            cur_status = lbl[len(status_marker_prefix):]
            status_of[cur_sid] = cur_status
        r["step_id"] = cur_sid if cur_sid >= 0 else None
    return status_of


def derive_status(rows, marker_prefix="target_spec_wait_"):
    """Map step_id -> status using target_spec_wait_<status> labels.

    Returns dict[step_id] -> status. For older JSONs without step_id,
    falls back to sequential assignment and writes step_id into rows.
    For sync SD (no status-suffixed labels), buckets everything under "all".
    """
    has_sid = any(r.get("step_id") is not None for r in rows)
    has_marker = any(r.get("label", "").startswith(marker_prefix) for r in rows)
    if has_sid:
        status_of = {}
        for r in rows:
            lbl = r.get("label", "")
            if lbl.startswith(marker_prefix):
                s = lbl[len(marker_prefix):]
                sid = r.get("step_id")
                if sid is not None:
                    status_of[sid] = s
        return status_of
    if has_marker:
        return _assign_sequential_step_ids(rows, marker_prefix)
    # No status marker — sync SD or non-spec run. Bucket all rows under "all".
    bare_marker = marker_prefix.rstrip("_")  # e.g. "target_spec_wait"
    cur_sid = -1
    status_of = {}
    rows_sorted = sorted(rows, key=lambda r: r.get("idx", 0)) if rows and "idx" in rows[0] else rows
    for r in rows_sorted:
        lbl = r.get("label", "")
        if lbl == bare_marker:
            cur_sid += 1
            status_of[cur_sid] = "all"
        r["step_id"] = cur_sid if cur_sid >= 0 else None
    return status_of


def aggregate(rows, status_of, events):
    """status -> event -> list of per-step summed durations (ms).

    For phase{1,2}_prep/replay (occurring K1/K2 times per step) we sum within
    the step so the bar shows total time per status step, not single-occurrence.
    For hit_cache_respond_<status> we strip the suffix.
    """
    per_step = defaultdict(lambda: defaultdict(float))  # (status, step_id) -> event -> sum
    for r in rows:
        sid = r.get("step_id")
        if sid is None:
            continue
        st = status_of.get(sid)
        if st is None:
            continue
        lbl = r.get("label", "")
        if lbl.startswith("hit_cache_respond_"):
            lbl = "hit_cache_respond"
        if lbl in events:
            per_step[(st, sid)][lbl] += _get_dur(r)

    bucket = defaultdict(lambda: defaultdict(list))
    for (st, sid), evdict in per_step.items():
        for ev, total in evdict.items():
            bucket[st][ev].append(total)
    return bucket


STATUS_LABELS = {
    "hit_k1": "Phase 1 hit",
    "hit_k2": "Phase 2 hit",
    "hit": "Hit",
    "miss": "Miss",
    "all": "All steps",
}
# Preferred ordering — MESA uses hit_k1/hit_k2/miss, SD uses hit/miss,
# sync/AR uses "all" (single bucket).
STATUS_ORDER = ["hit_k1", "hit_k2", "hit", "miss", "all"]


def stack_bar(ax, bucket, events, colors, title, y_label="mean cuda_ms"):
    statuses = [s for s in STATUS_ORDER if s in bucket]
    x = np.arange(len(statuses))
    bottoms = np.zeros(len(statuses))
    means_matrix = []
    for ev in events:
        row = []
        for st in statuses:
            vals = bucket[st].get(ev, [])
            row.append(float(np.mean(vals)) if vals else 0.0)
        means_matrix.append(row)

    # Collect small-segment callouts per status column for outside placement.
    small_per_status = [[] for _ in statuses]

    for ev_idx, ev in enumerate(events):
        means = np.array(means_matrix[ev_idx])
        if means.sum() == 0:
            continue
        ax.bar(x, means, bottom=bottoms, label=ev,
               color=colors.get(ev, "#cccccc"), edgecolor="white", linewidth=0.5)
        for xi, (m, b) in enumerate(zip(means, bottoms)):
            if m <= 0:
                continue
            if m < 1.0:
                # too small for in-bar label — defer to outside callout
                small_per_status[xi].append((b + m / 2, m, colors.get(ev, "#666666")))
            else:
                ax.text(xi, b + m / 2, f"{m:.2f}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="medium")
        bottoms += means

    # Place outside callouts for small segments with vertical stagger.
    max_y = float(max(bottoms)) if len(bottoms) else 0.0
    min_spacing = max_y * 0.038  # min vertical gap between stacked callouts
    for xi, segs in enumerate(small_per_status):
        if not segs:
            continue
        segs.sort()  # by y_center
        y_positions = [s[0] for s in segs]
        # Push down to ensure spacing
        for i in range(1, len(y_positions)):
            if y_positions[i] - y_positions[i - 1] < min_spacing:
                y_positions[i] = y_positions[i - 1] + min_spacing
        for (y_seg, m, color), y_text in zip(segs, y_positions):
            x_bar_edge = xi + 0.4
            x_text = xi + 0.43
            ax.plot([x_bar_edge, x_text - 0.005], [y_seg, y_text],
                    color="gray", lw=0.6, alpha=0.7)
            ax.text(x_text, y_text, f"{m:.2f}", ha="left", va="center",
                    fontsize=8, color=color, fontweight="bold")

    labels = [STATUS_LABELS.get(st, st) for st in statuses]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for xi, total in enumerate(bottoms):
        ax.text(xi, total + max_y * 0.012, f"Σ {total:.2f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")


def write_csv(bucket, events, path):
    statuses = [s for s in STATUS_ORDER if s in bucket]
    rows = []
    for st in statuses:
        n = len(next(iter(bucket[st].values()), []))
        row = {"status": st, "n": n}
        for ev in events:
            vals = bucket[st].get(ev, [])
            row[f"{ev}_mean_ms"] = float(np.mean(vals)) if vals else 0.0
            row[f"{ev}_median_ms"] = float(np.median(vals)) if vals else 0.0
        rows.append(row)
    import csv
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    print(f"OUTDIR = {OUTDIR}")
    t_rows = load_rows("mesa_profile_target_rank0_*.json")
    try:
        d_rows = load_rows("mesa_profile_draft_*.json")
    except FileNotFoundError:
        d_rows = None  # sync mode has no separate draft JSON
    status_of = derive_status(t_rows, "target_spec_wait_")
    d_status_of = None
    if d_rows is not None:
        d_has_sid = any(r.get("step_id") is not None for r in d_rows)
        if not d_has_sid:
            d_status_of = _assign_sequential_step_ids(d_rows, "hit_cache_respond_")
        else:
            d_status_of = status_of
    d_count = len(d_rows) if d_rows else 0
    print(f"target rows={len(t_rows):,}  draft rows={d_count:,}  steps={len(status_of):,}")

    # --- target ---
    t_bucket = aggregate(t_rows, status_of, set(TARGET_EVENTS))
    fig, ax = plt.subplots(figsize=(7.5, 7))
    stack_bar(ax, t_bucket, TARGET_EVENTS, TARGET_COLORS,
              "DUET target breakdown by status",
              y_label="target latency (ms)")
    fig.tight_layout()
    out_png = OUTDIR / "target_breakdown_by_status.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_png}")
    write_csv(t_bucket, TARGET_EVENTS, OUTDIR / "target_breakdown_by_status.csv")
    print(f"-> target_breakdown_by_status.csv")

    # --- draft (skip if no separate draft JSON, e.g. sync SD) ---
    if d_rows is not None and d_status_of is not None:
        d_bucket = aggregate(d_rows, d_status_of, set(DRAFT_EVENTS))
        fig, ax = plt.subplots(figsize=(7.5, 7))
        stack_bar(ax, d_bucket, DRAFT_EVENTS, DRAFT_COLORS,
                  "DUET draft breakdown by status",
                  y_label="draft latency (ms)")
        fig.tight_layout()
        out_png = OUTDIR / "draft_breakdown_by_status.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out_png}")
        write_csv(d_bucket, DRAFT_EVENTS, OUTDIR / "draft_breakdown_by_status.csv")
        print(f"-> draft_breakdown_by_status.csv")
    else:
        d_bucket = None
        print("(skip draft breakdown — no draft JSON, likely sync mode)")

    # --- summary table to stdout ---
    print("\n=== TARGET mean ms by status ===")
    for st in STATUS_ORDER:
        if st not in t_bucket:
            continue
        total = sum(np.mean(v) for v in t_bucket[st].values() if v)
        print(f"  {st:8s} total={total:6.2f} ms  n={len(next(iter(t_bucket[st].values()), [])):,}")

    if d_bucket is not None:
        print("\n=== DRAFT mean ms by status ===")
        for st in STATUS_ORDER:
            if st not in d_bucket:
                continue
            total = sum(np.mean(v) for v in d_bucket[st].values() if v)
            print(f"  {st:8s} total={total:6.2f} ms  n={len(next(iter(d_bucket[st].values()), [])):,}")


if __name__ == "__main__":
    main()
