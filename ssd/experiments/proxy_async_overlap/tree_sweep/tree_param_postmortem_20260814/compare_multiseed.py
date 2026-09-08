#!/usr/bin/env python3
"""Aggregate a balanced N2 sweep across sampler seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

from compare_arms import (
    load_question_module,
    question_rows,
    read_rows,
    step_weighted,
)


METRICS = (
    "tps",
    "accept_len",
    "cache_hit_rate",
    "p1_hit_rate",
    "p1_accept_len",
    "p2_hit_rate",
    "p2_accept_len",
)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def mean_sd(values: list[float]) -> str:
    return f"{fmean(values):.3f} ± {stdev(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="SEED,N2,PATH",
        help="repeat once per completed arm",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    qmod = load_question_module()
    by_seed: dict[int, dict[int, dict]] = defaultdict(dict)
    for spec in args.arm:
        seed_text, n2_text, path_text = spec.split(",", 2)
        seed, n2 = int(seed_text), int(n2_text)
        path = Path(path_text)
        raw_path = path / "raw.jsonl" if path.is_dir() else path
        raw = read_rows(raw_path)
        metric = qmod.aggregate(raw_path)
        metric["target_step_ms"] = step_weighted(raw, "mean_target_step_s")
        metric["verify_ms"] = step_weighted(raw, "mean_target_verify_s")
        metric["question_rows"] = question_rows(qmod, raw)
        by_seed[seed][n2] = metric

    expected_n2 = {10, 11, 12}
    for seed, arms in by_seed.items():
        if set(arms) != expected_n2:
            raise ValueError(f"seed {seed} has N2={sorted(arms)}, expected 10/11/12")
        keys = [set(arms[n]["question_rows"]) for n in sorted(arms)]
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError(f"question mismatch within seed {seed}")

    lines = [
        "# P2 N2/M2 multiseed comparison",
        "",
        "> Fixed 60 questions/70 turns, output cap 256, profiler off. "
        "All arms use K1/K2=8/5, exit 49, P1 N/M=14/12, and P2 M2=10.",
        "",
        "## Per seed",
        "",
        "| Seed | N2/M2 | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted(by_seed):
        for n2 in sorted(by_seed[seed]):
            m = by_seed[seed][n2]
            lines.append(
                f"| {seed} | {n2}/10 | {fmt(m['tps'])} | "
                f"{fmt(m['accept_len'])} | {fmt(m['cache_hit_rate'])} | "
                f"{fmt(m['p1_hit_rate'])} | {fmt(m['p1_accept_len'])} | "
                f"{fmt(m['p2_hit_rate'])} | {fmt(m['p2_accept_len'])} | "
                f"{fmt(m['target_step_ms'])} | {fmt(m['verify_ms'])} |"
            )

    lines += [
        "",
        "## Seed mean and variability",
        "",
        "| N2/M2 | TPS | AL | Hit | P1 AL | P2 AL | Target step (ms) | Verify (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n2 in sorted(expected_n2):
        vals = [by_seed[s][n2] for s in sorted(by_seed)]
        lines.append(
            f"| {n2}/10 | {mean_sd([m['tps'] for m in vals])} | "
            f"{mean_sd([m['accept_len'] for m in vals])} | "
            f"{mean_sd([m['cache_hit_rate'] for m in vals])} | "
            f"{mean_sd([m['p1_accept_len'] for m in vals])} | "
            f"{mean_sd([m['p2_accept_len'] for m in vals])} | "
            f"{mean_sd([m['target_step_ms'] for m in vals])} | "
            f"{mean_sd([m['verify_ms'] for m in vals])} |"
        )

    lines += [
        "",
        "## Paired direction versus N2=10",
        "",
        "| Seed | Candidate | mean ΔTPS | TPS wins | mean ΔAL | AL wins |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted(by_seed):
        ref = by_seed[seed][10]["question_rows"]
        for n2 in (11, 12):
            candidate = by_seed[seed][n2]["question_rows"]
            dtps, dal = [], []
            for key in sorted(ref):
                if ref[key]["tps"] is not None and candidate[key]["tps"] is not None:
                    dtps.append(candidate[key]["tps"] - ref[key]["tps"])
                if (
                    ref[key]["accept_len"] is not None
                    and candidate[key]["accept_len"] is not None
                ):
                    dal.append(candidate[key]["accept_len"] - ref[key]["accept_len"])
            lines.append(
                f"| {seed} | {n2}/10 | {fmt(fmean(dtps))} | "
                f"{sum(x > 0 for x in dtps)}/{len(dtps)} | {fmt(fmean(dal))} | "
                f"{sum(x > 0 for x in dal)}/{len(dal)} |"
            )

    lines += ["", "## Per-subtask seed means", ""]
    for n2 in sorted(expected_n2):
        lines += [
            f"### N2/M2={n2}/10",
            "",
            "| Subtask | TPS | AL | Hit | P1 AL | P2 AL |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for group in qmod.GROUPS:
            rows = [by_seed[s][n2]["groups"][group] for s in sorted(by_seed)]
            lines.append(
                f"| {group} | {fmt(fmean(r['tps'] for r in rows))} | "
                f"{fmt(fmean(r['accept_len'] for r in rows))} | "
                f"{fmt(fmean(r['cache_hit_rate'] for r in rows))} | "
                f"{fmt(fmean(r['p1_accept_len'] for r in rows))} | "
                f"{fmt(fmean(r['p2_accept_len'] for r in rows))} |"
            )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
