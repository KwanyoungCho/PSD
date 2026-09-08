#!/usr/bin/env python3
"""Render one profiled tree arm into a compact postmortem table."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def q(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", type=Path)
    ap.add_argument("--k1", type=int, required=True)
    ap.add_argument("--k2", type=int, required=True)
    ap.add_argument("--rpp", type=int, default=3)
    ap.add_argument("--p2-roots", type=int, default=10)
    args = ap.parse_args()
    arm = args.arm
    metric = json.load((arm / "metrics.json").open())[0]
    outcome = json.load((arm / "tree_outcomes.json").open())
    widths = json.load((arm / "round_widths.json").open())
    overlap = json.load((arm / "overlap.json").open())["runs"][0]
    serves = [json.loads(line) for line in (arm / "topology.serve.jsonl").open()
              if line.strip()]

    lines = [
        f"# {arm.name}", "",
        "> Diagnostic run: profiling and topology tracing were enabled; its TPS "
        "is not a performance claim.", "",
        "## Outcome", "",
        "| Questions (turns) | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metric['questions']} ({metric['turns']}) | "
        f"{fmt(metric['accept_len'])} | {fmt(metric['cache_hit_rate'])} | "
        f"{fmt(metric['p1_hit_rate'])} | {fmt(metric['p1_accept_len'])} | "
        f"{fmt(metric['p2_hit_rate'])} | {fmt(metric['p2_accept_len'])} |",
        "", "## Tree opportunity", "",
        "| Phase | Hit trees | Accepted nodes/tree | Reaches max depth | "
        "Alternative-sibling tree rate | Branch-assisted accepted share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, depth in (("p1", args.k1), ("p2", args.k2)):
        o = outcome["by_phase"][phase]
        w = widths[phase]
        reaches = w["accept_reaches_depth_rate"].get(str(depth), 0.0)
        lines.append(
            f"| {phase.upper()} | {o['trees']} | "
            f"{o['accepted_nodes_per_tree']:.3f} | {reaches:.3%} | "
            f"{o['alternative_tree_rate']:.3%} | "
            f"{o['branch_assisted_share_of_accepted']:.3%} |")

    lines += ["", "## Overlap tails", "",
              "Positive signed gap means the draft finished before its deadline.",
              "",
              "| Phase | aligned steps | late rate | signed p01 | signed p05 | "
              "signed p50 | overrun p95 | overrun p99 | max overrun |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for phase in ("k1", "k2"):
        row = overlap[f"{phase}_gap"]
        lines.append(
            f"| {phase.upper()} | {row['n']} | {row['late_rate']:.3%} | "
            f"{row['p01_ms']:.3f} | {row['p05_ms']:.3f} | "
            f"{row['p50_ms']:.3f} | {row['overrun_p95_ms']:.3f} | "
            f"{row['overrun_p99_ms']:.3f} | {row['overrun_max_ms']:.3f} |")

    lines += ["", "## Served-root pressure", "",
              "| Phase | serves | rank p50 | rank p90 | rank p95 | max | "
              "boundary-tail rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for phase, boundary in ((1, None), (2, args.p2_roots)):
        ranks = [int(x["root_rank"]) for x in serves if int(x["phase"]) == phase]
        if phase == 1:
            # P1's global rank is context_id * roots_per_position + local
            # token rank.  Budget pressure is whether the last local token
            # candidate is useful, not whether a hit happened in the final
            # context row.
            pressure_ranks = [x % args.rpp for x in ranks]
            boundary = args.rpp
            pressure_floor = boundary - 1
        else:
            pressure_ranks = ranks
            pressure_floor = boundary - 2
        tail = (sum(x >= pressure_floor for x in pressure_ranks) / len(ranks)
                if ranks else 0)
        lines.append(
            f"| P{phase} | {len(ranks)} | {fmt(q(ranks, .5), 1)} | "
            f"{fmt(q(ranks, .9), 1)} | {fmt(q(ranks, .95), 1)} | "
            f"{max(ranks) if ranks else '—'} | {tail:.3%} |")

    p1_context = Counter(int(x["root_context_id"]) for x in serves
                         if int(x["phase"]) == 1 and
                         x.get("root_context_id") is not None)
    p1_local_rank = Counter(int(x["root_rank"]) % args.rpp for x in serves
                            if int(x["phase"]) == 1)
    lines += ["", "Boundary tail is P1's last local token rank and P2's "
              "last two configured root ranks.", "",
              "P1 hit context counts: `" + json.dumps(
        dict(sorted(p1_context.items()))) + "`", "",
        "P1 local root-rank counts: `" + json.dumps(
        dict(sorted(p1_local_rank.items()))) + "`", ""]
    text = "\n".join(lines)
    (arm / "SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
