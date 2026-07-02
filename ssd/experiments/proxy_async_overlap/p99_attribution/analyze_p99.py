#!/usr/bin/env python3
"""Final-Analyze — target_spec_wait p99 attribution.

Consumes a Phase-B aligned JSON pair (target rank 0 + draft) and produces:

  1. Per-status target_spec_wait p50 / p90 / p99 (mean + count + total).
  2. Top-N p99 outlier step_ids per status (hit_k1 / hit_k2 / miss).
  3. For each top step, a draft-side attribution table: which draft labels
     overlapped target's `[target_send_request.start, target_response_received.start]`
     interval, with overlap duration in ms.
  4. Aligned timeline PNGs for representative p99 step per status
     (via plot_duet_aligned_timeline).
  5. RESULTS.md with all of the above.

Usage:
    python analyze_p99.py \\
        --target-json full/duet_profile_target_rank0_*.json \\
        --draft-json  full/duet_profile_draft_*.json \\
        --outdir      full \\
        --top-n 5

If --target-json / --draft-json are omitted, the latest duet_profile_*.json / mesa_profile_*.json
in --outdir is used.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Allow importing the plotter from bench/.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
import plot_duet_aligned_timeline as aligned  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _latest(outdir: Path, tag: str) -> Path:
    # Accepts both new duet_profile_*.json and legacy mesa_profile_*.json
    paths = sorted(outdir.glob(f"duet_profile_{tag}_*.json"))
    if not paths:
        paths = sorted(outdir.glob(f"mesa_profile_{tag}_*.json"))
    if not paths:
        raise FileNotFoundError(f"no duet_profile_{tag}_*.json under {outdir}")
    return paths[-1]


def load(target_path: Path, draft_path: Path) -> tuple[list[dict], list[dict]]:
    t = json.loads(target_path.read_text())
    d = json.loads(draft_path.read_text())
    if not aligned.is_aligned_schema(t):
        raise ValueError(f"{target_path} is not Phase-B aligned schema")
    if not aligned.is_aligned_schema(d):
        raise ValueError(f"{draft_path} is not Phase-B aligned schema")
    return aligned._strip_anchor(t), aligned._strip_anchor(d)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    k = min(n - 1, int(q * n))
    return xs_sorted[k]


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "p99": float("nan"),
                "total_ms": 0.0, "max": float("nan")}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "p50": statistics.median(xs),
        "p90": _percentile(xs, 0.90),
        "p99": _percentile(xs, 0.99),
        "max": max(xs),
        "total_ms": sum(xs),
    }


def target_spec_wait_by_status(target_rows: list[dict]) -> dict[str, list[dict]]:
    """Return {status: [span row, ...]} for spans labeled `target_spec_wait*`.

    Both raw status field (set via duet_set_context) and the legacy label
    suffix are honored, with status field taking precedence.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for r in target_rows:
        label = r.get("label") or ""
        if not label.startswith("target_spec_wait"):
            continue
        status = r.get("status")
        if not status:
            # Fall back to label suffix: target_spec_wait_hit_k1 → hit_k1
            if label == "target_spec_wait":
                status = "unknown"
            else:
                status = label[len("target_spec_wait_"):]
        out[status].append(r)
    return out


# ---------------------------------------------------------------------------
# Top-N outlier selection
# ---------------------------------------------------------------------------

def top_n_outliers(spans: list[dict], n: int = 5) -> list[dict]:
    """Return the n spans with the largest cuda_ms."""
    def _dur(r): return float(r.get("cuda_ms") or 0.0)
    return sorted(spans, key=_dur, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Draft attribution — which draft labels overlap target's wait window?
# ---------------------------------------------------------------------------

def find_target_wait_window(target_rows: list[dict], step_id: int) -> tuple[int, int] | None:
    """The wait window for step_id is bounded by target_send_request.start →
    target_response_received.start (or _.end if absent).

    Returns (start_ns, end_ns) in wall_start_ns units, or None if markers
    are missing.
    """
    by_label = defaultdict(list)
    for r in target_rows:
        if r.get("step_id") == step_id:
            by_label[r.get("label")].append(r)
    if not by_label.get("target_send_request") or not by_label.get("target_response_received"):
        # Fallback: use target_spec_wait_* span boundaries directly.
        sw = [r for r in target_rows
              if r.get("step_id") == step_id
              and (r.get("label") or "").startswith("target_spec_wait")]
        if not sw:
            return None
        s = sw[0]
        return (int(s["wall_start_ns"]), int(s["wall_end_ns"]))
    s_start = int(by_label["target_send_request"][0]["wall_start_ns"])
    s_end = int(by_label["target_response_received"][0]["wall_start_ns"])
    return (s_start, s_end)


def overlap_ns(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0, hi - lo)


def attribute_draft_activity(
    target_rows: list[dict],
    draft_rows: list[dict],
    step_id: int,
) -> dict:
    """For the target wait window of `step_id`, compute draft overlap per label."""
    win = find_target_wait_window(target_rows, step_id)
    if win is None:
        return {"window_ns": None, "window_ms": 0.0, "overlaps": []}
    win_start, win_end = win
    window_ms = (win_end - win_start) / 1e6

    bucket: dict[tuple[str, int | None], int] = defaultdict(int)
    bucket_step: dict[tuple[str, int | None], int] = defaultdict(int)
    for r in draft_rows:
        ws = r.get("wall_start_ns")
        we = r.get("wall_end_ns")
        if ws is None or we is None:
            continue
        ov = overlap_ns(win_start, win_end, int(ws), int(we))
        if ov <= 0:
            continue
        key = (r.get("label") or "", r.get("step_id"))
        bucket[key] += ov

    rows = []
    for (label, ds), ov in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "label": label,
            "draft_step_id": ds,
            "step_id_offset": (ds - step_id) if (isinstance(ds, int)) else None,
            "overlap_ms": ov / 1e6,
            "share_pct": (ov / (win_end - win_start) * 100) if win_end > win_start else 0.0,
        })
    return {
        "window_ns": win,
        "window_ms": window_ms,
        "overlaps": rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_stats(name: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{name:30s}  n=0"
    return (
        f"{name:30s}  n={s['n']:5d}  mean={s['mean']:6.3f}  "
        f"p50={s['p50']:6.3f}  p90={s['p90']:6.3f}  p99={s['p99']:6.3f}  "
        f"max={s['max']:7.3f}  total={s['total_ms']:9.2f} ms"
    )


def render_attribution_table(report: dict, max_rows: int = 10) -> str:
    """Markdown table for one step's attribution."""
    if report["window_ns"] is None:
        return "_(no markers; window unresolved)_"
    lines = []
    lines.append(
        f"window: {report['window_ms']:.3f} ms "
        f"(wall ns {report['window_ns'][0]} → {report['window_ns'][1]})"
    )
    lines.append("")
    lines.append("| draft label | draft step_id | Δstep | overlap (ms) | share (%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["overlaps"][:max_rows]:
        off = row["step_id_offset"]
        off_s = f"{off:+d}" if isinstance(off, int) else "n/a"
        lines.append(
            f"| `{row['label']}` | {row['draft_step_id']} | {off_s} | "
            f"{row['overlap_ms']:.3f} | {row['share_pct']:.1f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-json", type=Path)
    ap.add_argument("--draft-json", type=Path)
    ap.add_argument("--outdir", type=Path, required=True,
                    help="dir for outputs (RESULTS.md, PNGs). Also the default search dir for the JSONs.")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--png-statuses", nargs="*", default=["hit_k1", "hit_k2", "miss"])
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    t_path = args.target_json or _latest(args.outdir, "target_rank0")
    d_path = args.draft_json or _latest(args.outdir, "draft")

    target, draft = load(t_path, d_path)
    print(f"loaded target rows: {len(target)} from {t_path.name}")
    print(f"loaded draft  rows: {len(draft)} from {d_path.name}")

    by_status = target_spec_wait_by_status(target)

    # Per-status stats
    print("\n--- target_spec_wait per status ---")
    stats_by_status: dict[str, dict] = {}
    for status in sorted(by_status.keys()):
        durs = [float(r.get("cuda_ms") or 0.0) for r in by_status[status]]
        s = stats(durs)
        stats_by_status[status] = s
        print(fmt_stats(f"target_spec_wait[{status}]", s))

    # Top-N outliers per status
    print(f"\n--- top-{args.top_n} cuda_ms outliers per status ---")
    top_by_status: dict[str, list[dict]] = {}
    for status, spans in by_status.items():
        tops = top_n_outliers(spans, args.top_n)
        top_by_status[status] = tops
        print(f"\n[{status}]")
        for r in tops:
            print(f"  step_id={r.get('step_id'):4d}  cuda_ms={r.get('cuda_ms'):7.3f}  "
                  f"wall_start_ns={r.get('wall_start_ns')}")

    # Attribution per top step
    attribution: dict[str, list[dict]] = {}
    for status, tops in top_by_status.items():
        attribution[status] = []
        for r in tops:
            sid = r.get("step_id")
            if sid is None:
                continue
            rep = attribute_draft_activity(target, draft, sid)
            rep["step_id"] = sid
            rep["target_spec_wait_ms"] = float(r.get("cuda_ms") or 0.0)
            attribution[status].append(rep)

    # PNGs — representative p99 step (= the worst step of that status; first in top-N).
    png_paths: dict[str, Path] = {}
    for status in args.png_statuses:
        tops = top_by_status.get(status, [])
        if not tops:
            print(f"\n[png:{status}] no spans, skip")
            continue
        sid = tops[0].get("step_id")
        if sid is None:
            continue
        out_png = args.outdir / f"timeline_p99_{status}_step{sid}.png"
        try:
            tr, dr = aligned.select_step(target, draft, sid)
            aligned.plot_aligned_step(tr, dr, sid, out_png, title_suffix=f"(p99 {status})")
            png_paths[status] = out_png
            print(f"[png:{status}] step_id={sid} -> {out_png}")
        except Exception as e:
            print(f"[png:{status}] step_id={sid} failed: {e}")

    # ----- RESULTS.md -----
    md = []
    md.append("# Final p99 attribution — target_spec_wait")
    md.append("")
    md.append(f"**Target JSON**: `{t_path.name}` ({len(target)} spans)")
    md.append(f"**Draft JSON**:  `{d_path.name}` ({len(draft)} spans)")
    md.append("")
    md.append("## 1. Per-status target_spec_wait stats")
    md.append("")
    md.append("| status | n | mean | p50 | p90 | p99 | max | total (ms) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for status, s in sorted(stats_by_status.items()):
        md.append(
            f"| {status} | {s['n']} | {s['mean']:.3f} | {s['p50']:.3f} | "
            f"{s['p90']:.3f} | {s['p99']:.3f} | {s['max']:.3f} | {s['total_ms']:.2f} |"
        )
    md.append("")
    md.append("## 2. Top-N target_spec_wait outliers and draft attribution")
    md.append("")
    for status in sorted(top_by_status.keys()):
        md.append(f"### status = {status}")
        if status in png_paths:
            md.append(f"![{status}]({png_paths[status].name})")
        md.append("")
        for entry in attribution.get(status, []):
            md.append(f"**step_id={entry['step_id']}  target_spec_wait={entry['target_spec_wait_ms']:.3f} ms**")
            md.append("")
            md.append(render_attribution_table(entry))
            md.append("")
    out_md = args.outdir / "RESULTS.md"
    out_md.write_text("\n".join(md))
    print(f"\nwrote {out_md}")

    # JSON for machine consumption
    out_json = args.outdir / "attribution.json"
    out_json.write_text(json.dumps({
        "stats_by_status": stats_by_status,
        "top_by_status": {
            s: [{"step_id": r.get("step_id"),
                 "cuda_ms": r.get("cuda_ms"),
                 "wall_start_ns": r.get("wall_start_ns")} for r in tops]
            for s, tops in top_by_status.items()
        },
        "attribution": {s: items for s, items in attribution.items()},
        "png_paths": {s: str(p) for s, p in png_paths.items()},
        "target_json": str(t_path),
        "draft_json": str(d_path),
    }, indent=2, default=str))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
