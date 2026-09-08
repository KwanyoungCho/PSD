#!/usr/bin/env python3
"""Separate fixed tree machinery from the cost of additional verify rows."""
from __future__ import annotations

import csv
import importlib.util
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTROL = HERE / "matched_width_controls"
TRIAD = HERE / "full_gap_triad"
spec = importlib.util.spec_from_file_location(
    "triad_analysis", HERE / "analyze_full_gap_triad.py")
triad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triad)


def complete_paths(root):
    return [path for path in sorted(root.glob("*_r*_s42_o256.jsonl"))
            if sum(1 for line in path.open() if line.strip()) == 7
            and list(path.with_name(path.stem + "_profile").glob("*target*.json"))]


def load(root):
    overalls, summaries = [], []
    for path in complete_paths(root):
        overall, per_status = triad.summarize_run(path)
        overalls.append(overall); summaries.extend(per_status)
    return overalls, summaries


def agg(rows, arm, status, metric):
    vals = [row[metric] for row in rows
            if row["arm"] == arm and row["status"] == status
            and not math.isnan(row[metric])]
    if not vals:
        return math.nan, 0.0
    return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def fmt(value):
    mean, spread = value
    return "—" if math.isnan(mean) else f"{mean:.3f} ± {spread:.3f}"


def diff(rows, left, right, status, metric):
    l = agg(rows, left, status, metric)[0]
    r = agg(rows, right, status, metric)[0]
    return r - l


def make_table(lines, title, status, arms, metrics, summaries):
    lines += [f"### {title}", "",
              "| Segment | " + " | ".join(label for label, _ in arms) + " |",
              "|---|" + "---:|" * len(arms)]
    for label, metric in metrics:
        vals = [fmt(agg(summaries, arm, status, metric)) for _, arm in arms]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")


def main():
    triad_overall, triad_summaries = load(TRIAD)
    control_overall, control_summaries = load(CONTROL)
    overalls = triad_overall + control_overall
    summaries = triad_summaries + control_summaries
    if not control_overall:
        raise SystemExit("no completed matched-width controls")

    fields = sorted({key for row in summaries for key in row})
    with (CONTROL / "conditional_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)

    critical = (
        ("Full step", "target_full_profile_ms"),
        ("Draft/spec wait", "target_spec_wait_ms"),
        ("Pre-verify", "target_preverify_gap_ms"),
        ("Target verify", "target_verify_profile_ms"),
    )
    preverify = (
        ("Wire parse/validate", "target_wire_validate_ms"),
        ("Topology CPU pack", "target_topology_cpu_pack_ms"),
        ("Topology H2D", "target_topology_h2d_ms"),
        ("Parent-q select", "target_parent_q_select_ms"),
    )
    verify = (
        ("Verify setup", "verify_setup_ms"),
        ("Graph pre", "verify_graph_pre_ms"),
        ("Exit/proxy side", "verify_exit_proxy_ms"),
        ("Graph post", "verify_graph_post_ms"),
        ("Acceptance envelope", "verify_sample_accept_ms"),
    )
    p1_arms = (("Chain, 8 nodes / 9 rows", "chain"),
               ("Tree, 8 nodes / 9 rows", "p1_tree8"),
               ("Tree, 12 nodes / 13 rows", "p1_tree12"))
    p2_arms = (("Chain, 4 nodes / 5 rows", "chain"),
               ("Tree, 4 nodes / 5 rows", "p2_tree4"),
               ("Tree, 8 nodes / 9 rows", "p2_tree"))

    lines = [
        "# Matched-width tree latency controls",
        "",
        "These latency-only controls keep the number of target verification "
        "rows equal between chain and tree, then increase only the tree node "
        "count. They are not AL or hit-rate evaluations.",
        "",
        "- P1: chain K1=8 (9 rows), tree M1=8 (9 rows), tree M1=12 (13 rows).",
        "- P2: chain K2=4 (5 rows), tree M2=4 (5 rows), tree M2=8 (9 rows).",
        "",
        "## Critical path", "",
    ]
    make_table(lines, "P1 hit", "hit_k1", p1_arms, critical, summaries)
    make_table(lines, "P2 hit", "hit_k2", p2_arms, critical, summaries)
    lines += ["## Pre-verify fixed machinery", ""]
    make_table(lines, "P1 hit", "hit_k1", p1_arms, preverify, summaries)
    make_table(lines, "P2 hit", "hit_k2", p2_arms, preverify, summaries)
    lines += ["## Verify internals", ""]
    make_table(lines, "P1 hit", "hit_k1", p1_arms, verify, summaries)
    make_table(lines, "P2 hit", "hit_k2", p2_arms, verify, summaries)

    lines += [
        "## Causal split",
        "",
        "| Comparison | Full step | Draft wait | Pre-verify | Verify |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, status, left, right in (
        ("P1 fixed tree machinery: tree8 − chain8",
         "hit_k1", "chain", "p1_tree8"),
        ("P1 four extra nodes: tree12 − tree8",
         "hit_k1", "p1_tree8", "p1_tree12"),
        ("P2 fixed tree machinery: tree4 − chain4",
         "hit_k2", "chain", "p2_tree4"),
        ("P2 four extra nodes: tree8 − tree4",
         "hit_k2", "p2_tree4", "p2_tree"),
    ):
        vals = [diff(summaries, left, right, status, metric) for _, metric in critical]
        lines.append(f"| {label} | " + " | ".join(f"{v:+.3f} ms" for v in vals) + " |")

    lines += [
        "",
        "`tree8 − chain8` / `tree4 − chain4` estimates fixed tree protocol, "
        "metadata, custom-attention setup, and tree-walk cost at equal target "
        "row count. `tree12 − tree8` / `tree8 − tree4` estimates the marginal "
        "cost of four additional verification nodes, but generation paths and "
        "accepted outputs can still change; use status-conditional spans, not "
        "overall TPS, for this diagnostic.",
        "",
        "Machine-readable: `conditional_runs.csv`.",
    ]
    (CONTROL / "MATCHED_WIDTH_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(CONTROL / "MATCHED_WIDTH_ANALYSIS.md")


if __name__ == "__main__":
    main()
