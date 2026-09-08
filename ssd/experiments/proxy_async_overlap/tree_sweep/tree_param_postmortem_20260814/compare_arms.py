#!/usr/bin/env python3
"""Compare profiler-off DUET arms with paper and step-weighted conventions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import fmean
from typing import Any


QUESTION_METRICS = Path(
    "/home/eslab/chokwans99/baseline/analysis/question_level_metrics.py"
)


def load_question_module():
    spec = importlib.util.spec_from_file_location("question_level_metrics", QUESTION_METRICS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {QUESTION_METRICS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def step_weighted(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = []
    for row in rows:
        value = row.get(key)
        steps = row.get("n_verify_steps")
        if isinstance(value, (int, float)) and isinstance(steps, (int, float)) and steps > 0:
            pairs.append((float(value), float(steps)))
    if not pairs:
        return None
    return 1e3 * sum(value * steps for value, steps in pairs) / sum(
        steps for _, steps in pairs
    )


def question_rows(module, raw: list[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in raw:
        grouped.setdefault((row["group"], row["question_id"]), []).append(row)
    return {
        key: module.question_stats(sorted(rows, key=lambda row: row.get("turn", 0)))
        for key, rows in grouped.items()
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="Tree screening comparison")
    args = parser.parse_args()
    qmod = load_question_module()

    results = []
    questions = []
    for arm in args.arms:
        raw_path = arm / "raw.jsonl" if arm.is_dir() else arm
        raw = read_rows(raw_path)
        results.append((arm.name, qmod.aggregate(raw_path), raw))
        questions.append(question_rows(qmod, raw))

    reference_keys = set(questions[0])
    for (name, _, _), by_question in zip(results, questions):
        if set(by_question) != reference_keys:
            missing = sorted(reference_keys - set(by_question))
            extra = sorted(set(by_question) - reference_keys)
            raise ValueError(f"UID mismatch for {name}: missing={missing[:3]}, extra={extra[:3]}")

    lines = [
        f"# {args.title}",
        "",
        "> Profiler-off, fixed paired questions. TPS and AL are question-level; "
        "latencies are verification-step weighted.",
        "",
        "| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric, raw in results:
        lines.append(
            f"| {name} | {metric['questions']} ({metric['turns']}) | "
            f"{fmt(metric['tps'])} | {fmt(metric['accept_len'])} | "
            f"{fmt(metric['cache_hit_rate'])} | {fmt(metric['p1_hit_rate'])} | "
            f"{fmt(metric['p1_accept_len'])} | {fmt(metric['p2_hit_rate'])} | "
            f"{fmt(metric['p2_accept_len'])} | "
            f"{fmt(step_weighted(raw, 'mean_target_step_s'))} | "
            f"{fmt(step_weighted(raw, 'mean_target_verify_s'))} |"
        )

    ref = questions[0]
    lines += [
        "",
        "## Paired change from reference",
        "",
        "| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for (name, _, _), candidate in zip(results[1:], questions[1:]):
        tps_delta = []
        al_delta = []
        for key in sorted(reference_keys):
            rtps, ctps = ref[key]["tps"], candidate[key]["tps"]
            ral, cal = ref[key]["accept_len"], candidate[key]["accept_len"]
            if isinstance(rtps, (int, float)) and isinstance(ctps, (int, float)):
                tps_delta.append(ctps - rtps)
            if isinstance(ral, (int, float)) and isinstance(cal, (int, float)):
                al_delta.append(cal - ral)
        lines.append(
            f"| {name} | {fmt(fmean(tps_delta) if tps_delta else None)} | "
            f"{sum(x > 0 for x in tps_delta)}/{len(tps_delta)} | "
            f"{fmt(fmean(al_delta) if al_delta else None)} | "
            f"{sum(x > 0 for x in al_delta)}/{len(al_delta)} |"
        )

    lines += ["", "## Per-subtask", ""]
    for name, metric, _ in results:
        lines += [
            f"### {name}",
            "",
            "| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for group in qmod.GROUPS:
            row = metric["groups"][group]
            lines.append(
                f"| {group} | {row['questions']} ({row['turns']}) | "
                f"{fmt(row['tps'])} | {fmt(row['accept_len'])} | "
                f"{fmt(row['cache_hit_rate'])} | {fmt(row['p1_accept_len'])} | "
                f"{fmt(row['p2_accept_len'])} |"
            )
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
