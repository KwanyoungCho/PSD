#!/usr/bin/env python3
"""Phase 0b analyzer.

Reads mesa_profile_target_*.json from A_detail0 and B_detail1 runs, computes:

  - outer (proxy_compute_send) mean / median / p90 / p99
  - inner (proxy_compute / proxy_pack / proxy_send) mean (B only)
  - unattributed_stall_mean = outer_mean − Σ inner_mean (B only)
  - related target timings (exit_logits, graph_pre, graph_post,
    verify_sample_accept, target_spec_wait family)

Decision gates (printed at end):
  1. probe_effect: |A.outer_mean − B.outer_mean|
     < 0.3 ms  → safe to proceed
     ≥ 0.5 ms → STOP, calibration issue
  2. stall classification (B run)
     stall ≥ 1.0 ms ∧ send_mean < 0.3 ms  → "CPU-wait dominant" (Phase 2 expected to suffice)
     stall ≥ 1.0 ms ∧ send_mean ≥ 1.0 ms → "NCCL-GPU-time dominant" (Phase 3 required)
     stall < 0.5 ms                       → "no headroom" (STOP, revisit bottleneck)

Usage:
    python analyze.py
    python analyze.py --json-dir /custom/path
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import defaultdict


def load_target_profile(run_dir: pathlib.Path) -> dict[str, list[float]]:
    """Load mesa_profile_target_rank0_*.json and group ms by label."""
    candidates = sorted(run_dir.glob("mesa_profile_target_rank0_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no target profile in {run_dir}")
    # Pick newest
    path = candidates[-1]
    events = json.loads(path.read_text())
    groups: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        label = ev.get("label")
        ms = ev.get("ms")
        if label is None or ms is None:
            continue
        groups[str(label)].append(float(ms))
    return dict(groups)


def stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "p99": float("nan")}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    p90 = xs_sorted[min(n - 1, int(0.90 * n))]
    p99 = xs_sorted[min(n - 1, int(0.99 * n))]
    return {
        "n": n,
        "mean": statistics.fmean(xs_sorted),
        "median": statistics.median(xs_sorted),
        "p90": p90,
        "p99": p99,
    }


def fmt_stats(name: str, s: dict[str, float]) -> str:
    return (
        f"{name:32s}  n={s['n']:5d}  "
        f"mean={s['mean']:7.3f}  median={s['median']:7.3f}  "
        f"p90={s['p90']:7.3f}  p99={s['p99']:7.3f}"
    )


def sum_prefix(groups: dict[str, list[float]], prefix: str) -> list[float]:
    """Flatten all events whose label starts with prefix (e.g. target_spec_wait)."""
    out: list[float] = []
    for label, xs in groups.items():
        if label.startswith(prefix):
            out.extend(xs)
    return out


def analyze_run(label: str, run_dir: pathlib.Path) -> dict:
    print(f"\n========== RUN: {label}  ({run_dir.name}) ==========")
    groups = load_target_profile(run_dir)
    summary: dict[str, dict] = {}

    keys_top = [
        "graph_pre", "exit_logits", "proxy_compute_send",
        "graph_post", "final_logits", "verify_sample_accept", "verify_setup",
    ]
    for k in keys_top:
        s = stats(groups.get(k, []))
        summary[k] = s
        print(fmt_stats(k, s))

    # target_spec_wait family (sum across all suffixes)
    waits = sum_prefix(groups, "target_spec_wait")
    s_wait = stats(waits)
    summary["target_spec_wait_*"] = s_wait
    print(fmt_stats("target_spec_wait_*", s_wait))

    # Inner spans (DETAIL=1 only)
    inner_keys = ["proxy_compute", "proxy_pack", "proxy_send"]
    inner_means = []
    for k in inner_keys:
        if k in groups:
            s = stats(groups[k])
            summary[k] = s
            print(fmt_stats(k, s))
            inner_means.append(s["mean"])
    if inner_means:
        inner_sum_mean = sum(inner_means)
        outer_mean = summary["proxy_compute_send"]["mean"]
        stall_mean = outer_mean - inner_sum_mean
        summary["_inner_sum_mean"] = inner_sum_mean
        summary["_unattributed_stall_mean"] = stall_mean
        print(f"  → inner_sum_mean             = {inner_sum_mean:7.3f} ms")
        print(f"  → outer_mean                 = {outer_mean:7.3f} ms")
        print(f"  → unattributed_stall_mean    = {stall_mean:7.3f} ms")

    return summary


def decision(A: dict, B: dict) -> None:
    print("\n========== DECISION GATES ==========")
    a_outer = A["proxy_compute_send"]["mean"]
    b_outer = B["proxy_compute_send"]["mean"]
    delta = abs(a_outer - b_outer)
    print(f"A.outer_mean = {a_outer:.3f} ms")
    print(f"B.outer_mean = {b_outer:.3f} ms")
    print(f"|Δouter|     = {delta:.3f} ms")

    if delta < 0.3:
        probe_verdict = "GREEN  — DETAIL probe does not perturb outer."
    elif delta <= 0.5:
        probe_verdict = "YELLOW — proceed but note probe sensitivity."
    else:
        probe_verdict = "RED    — STOP. Probe perturbs outer; fix instrumentation before optimization."
    print(f"PROBE EFFECT: {probe_verdict}")

    # Stall classification from B
    if "_unattributed_stall_mean" in B:
        stall = B["_unattributed_stall_mean"]
        send = B.get("proxy_send", {}).get("mean", float("nan"))
        if stall >= 1.0 and send < 0.3:
            verdict = "CPU-wait dominant — Phase 2 (async send) likely sufficient."
        elif stall >= 1.0 and send >= 1.0:
            verdict = "NCCL-GPU-time dominant — Phase 3 (stream separation) required for full win."
        elif stall < 0.5:
            verdict = "RED — stall < 0.5 ms. STOP and revisit bottleneck source."
        else:
            verdict = "MIXED — moderate stall + moderate send. Phase 2 first, measure, decide Phase 3."
        print(f"STALL CLASS:  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent)
    args = ap.parse_args()

    A_dir = args.json_dir / "A_detail0"
    B_dir = args.json_dir / "B_detail1"

    if not A_dir.exists() or not B_dir.exists():
        print(f"Missing run dirs: {A_dir} / {B_dir}", file=sys.stderr)
        sys.exit(1)

    A = analyze_run("A_detail0", A_dir)
    B = analyze_run("B_detail1", B_dir)
    decision(A, B)


if __name__ == "__main__":
    main()
