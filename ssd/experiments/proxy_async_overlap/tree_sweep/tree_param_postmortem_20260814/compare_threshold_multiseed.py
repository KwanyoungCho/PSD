#!/usr/bin/env python3
"""Aggregate the current-tree P2 confidence-floor A/B across seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

from compare_arms import load_question_module, question_rows, read_rows, step_weighted


def ms(values: list[float]) -> str:
    return f"{fmean(values):.3f} ± {stdev(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True,
                        metavar="SEED,CONF,PATH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    qmod = load_question_module()
    data: dict[int, dict[float, dict]] = defaultdict(dict)
    for spec in args.arm:
        seed_text, conf_text, path_text = spec.split(",", 2)
        seed, conf = int(seed_text), float(conf_text)
        path = Path(path_text)
        raw_path = path / "raw.jsonl" if path.is_dir() else path
        raw = read_rows(raw_path)
        metric = qmod.aggregate(raw_path)
        metric["target_step_ms"] = step_weighted(raw, "mean_target_step_s")
        metric["verify_ms"] = step_weighted(raw, "mean_target_verify_s")
        metric["question_rows"] = question_rows(qmod, raw)
        data[seed][conf] = metric

    confs = {0.01, 0.02, 0.03}
    for seed, arms in data.items():
        if set(arms) != confs:
            raise ValueError(f"seed {seed}: thresholds={sorted(arms)}")
        keys = [set(arms[c]["question_rows"]) for c in sorted(confs)]
        if any(k != keys[0] for k in keys[1:]):
            raise ValueError(f"question mismatch in seed {seed}")

    lines = [
        "# P2 confidence threshold multiseed comparison", "",
        "> Fixed 60 questions/70 turns, output cap 256, profiler off. "
        "K1/K2=8/5, exit 49, P1 N/M=14/12, P2 N/M=10/10, "
        "and P2 proxy floor=0.01 are fixed.", "",
        "## Per seed", "",
        "| Seed | Confidence | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted(data):
        for conf in sorted(confs):
            m = data[seed][conf]
            lines.append(
                f"| {seed} | {conf:.2f} | {m['tps']:.3f} | "
                f"{m['accept_len']:.3f} | {m['cache_hit_rate']:.3f} | "
                f"{m['p1_hit_rate']:.3f} | {m['p1_accept_len']:.3f} | "
                f"{m['p2_hit_rate']:.3f} | {m['p2_accept_len']:.3f} | "
                f"{m['target_step_ms']:.3f} | {m['verify_ms']:.3f} |"
            )

    lines += ["", "## Seed mean and variability", "",
              "| Confidence | TPS | AL | Hit | P1 AL | P2 AL | Target step (ms) | Verify (ms) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for conf in sorted(confs):
        rows = [data[s][conf] for s in sorted(data)]
        lines.append(
            f"| {conf:.2f} | {ms([r['tps'] for r in rows])} | "
            f"{ms([r['accept_len'] for r in rows])} | "
            f"{ms([r['cache_hit_rate'] for r in rows])} | "
            f"{ms([r['p1_accept_len'] for r in rows])} | "
            f"{ms([r['p2_accept_len'] for r in rows])} | "
            f"{ms([r['target_step_ms'] for r in rows])} | "
            f"{ms([r['verify_ms'] for r in rows])} |"
        )

    lines += ["", "## Paired direction versus 0.01", "",
              "| Seed | Confidence | mean ΔTPS | TPS wins | mean ΔAL | AL wins |",
              "|---:|---:|---:|---:|---:|---:|"]
    for seed in sorted(data):
        ref = data[seed][0.01]["question_rows"]
        for conf in (0.02, 0.03):
            cand = data[seed][conf]["question_rows"]
            dtps, dal = [], []
            for key in sorted(ref):
                if ref[key]["tps"] is not None and cand[key]["tps"] is not None:
                    dtps.append(cand[key]["tps"] - ref[key]["tps"])
                if ref[key]["accept_len"] is not None and cand[key]["accept_len"] is not None:
                    dal.append(cand[key]["accept_len"] - ref[key]["accept_len"])
            lines.append(
                f"| {seed} | {conf:.2f} | {fmean(dtps):.3f} | "
                f"{sum(x > 0 for x in dtps)}/{len(dtps)} | "
                f"{fmean(dal):.3f} | {sum(x > 0 for x in dal)}/{len(dal)} |"
            )

    lines += ["", "## Per-subtask seed means", ""]
    for conf in sorted(confs):
        lines += [f"### Confidence={conf:.2f}", "",
                  "| Subtask | TPS | AL | Hit | P1 AL | P2 AL |",
                  "|---|---:|---:|---:|---:|---:|"]
        for group in qmod.GROUPS:
            rows = [data[s][conf]["groups"][group] for s in sorted(data)]
            lines.append(
                f"| {group} | {fmean(r['tps'] for r in rows):.3f} | "
                f"{fmean(r['accept_len'] for r in rows):.3f} | "
                f"{fmean(r['cache_hit_rate'] for r in rows):.3f} | "
                f"{fmean(r['p1_accept_len'] for r in rows):.3f} | "
                f"{fmean(r['p2_accept_len'] for r in rows):.3f} |"
            )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
