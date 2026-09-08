#!/usr/bin/env python3
"""Validate and summarize the three-seed full-data fused-rerank run."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/eslab/chokwans99/PSD/ssd")
BASE = Path("/home/eslab/chokwans99/baseline")
HERE = ROOT / (
    "experiments/proxy_async_overlap/tree_sweep/"
    "p1_p2_tree_full_rerank_3seed_20260812")
DATA = BASE / "data/specbench_full.jsonl"
SEEDS = (1, 42, 123)
GROUPS = ("mt_bench", "translation", "summarization", "qa",
          "math_reasoning", "rag")
QUESTION_METRICS = (
    "tps", "accept_len", "cache_hit_rate", "p1_hit_rate",
    "p1_accept_len", "p2_hit_rate", "p2_accept_len")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def weighted(rows: list[dict], key: str,
             weight_key: str = "n_verify_steps") -> float | None:
    pairs = []
    for row in rows:
        value, weight = number(row, key), number(row, weight_key)
        if value is not None and weight is not None and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return None
    return sum(value * weight for value, weight in pairs) / sum(
        weight for _, weight in pairs)


def phase_al(rows: list[dict], phase: str) -> float | None:
    pairs = []
    for row in rows:
        value = number(row, f"{phase}_accept_len")
        steps = number(row, "n_verify_steps")
        hit = number(row, f"{phase}_hit_rate")
        if value is not None and steps is not None and hit is not None:
            weight = steps * hit
            if weight > 0:
                pairs.append((value, weight))
    if not pairs:
        return None
    return sum(value * weight for value, weight in pairs) / sum(
        weight for _, weight in pairs)


def make_questions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["question_id"])].append(row)
    questions = []
    for (group, qid), turns in grouped.items():
        turns.sort(key=lambda row: row.get("turn", 0))
        tokens = sum(number(row, "decode_total_tokens") or 0 for row in turns)
        elapsed = sum(number(row, "decode_total_time") or 0 for row in turns)
        questions.append({
            "key": (group, qid), "group": group, "turns": len(turns),
            "tps": tokens / elapsed,
            "accept_len": weighted(turns, "accept_len"),
            "cache_hit_rate": weighted(turns, "cache_hit_rate"),
            "p1_hit_rate": weighted(turns, "p1_hit_rate"),
            "p1_accept_len": phase_al(turns, "p1"),
            "p2_hit_rate": weighted(turns, "p2_hit_rate"),
            "p2_accept_len": phase_al(turns, "p2"),
        })
    return questions


def mean_present(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows
              if isinstance(row.get(key), (int, float))]
    return statistics.fmean(values) if values else None


def step_ms(rows: list[dict], key: str) -> float | None:
    pairs = []
    for row in rows:
        value, steps = number(row, key), number(row, "n_verify_steps")
        if value is not None and steps is not None and steps > 0:
            pairs.append((value, steps))
    if not pairs:
        return None
    return 1000.0 * sum(value * steps for value, steps in pairs) / sum(
        steps for _, steps in pairs)


def aggregate(rows: list[dict]) -> dict:
    questions = make_questions(rows)
    result = {
        "questions": len(questions), "turns": len(rows),
        **{key: mean_present(questions, key) for key in QUESTION_METRICS},
        "target_step_ms": step_ms(rows, "mean_target_step_s"),
        "target_verify_ms": step_ms(rows, "mean_target_verify_s"),
        "verify_steps": sum(int(row["n_verify_steps"]) for row in rows),
        "completion_tokens": sum(
            int(row["decode_total_tokens"]) for row in rows),
    }
    result["outside_verify_ms"] = (
        result["target_step_ms"] - result["target_verify_ms"])
    result["groups"] = {}
    for group in GROUPS:
        selected = [row for row in questions if row["group"] == group]
        result["groups"][group] = {
            "questions": len(selected),
            "turns": sum(row["turns"] for row in selected),
            **{key: mean_present(selected, key) for key in QUESTION_METRICS},
        }
    return result


def validate(rows: list[dict], expected_uids: list[str], seed: int) -> None:
    if len(rows) != 560:
        raise RuntimeError(f"seed {seed}: {len(rows)}/560 rows")
    if [row.get("uid") for row in rows] != expected_uids:
        raise RuntimeError(f"seed {seed}: UID/order mismatch")
    common = {
        "engine": "duet-current(flashinfer,graph)",
        "k1": 8, "k2": 4, "exit_layer": 56,
        "p1_fanout": 3, "p2_budget": 15, "proxy_top_k": 28,
        "p1_tree": "on", "p2_tree": "on",
        "p1_allocation_policy": "backbone",
        "c_tensor": 2, "n1": 14, "n2": 8,
        "p1_verify_nodes": 12, "p2_verify_nodes": 8,
        "max_new_tokens": 1024, "max_model_len": 4096,
        "extend_draft_rope": True, "seed": seed, "sampler_seed": seed,
    }
    required = (
        "decode_total_tokens", "decode_total_time", "accept_len",
        "cache_hit_rate", "p1_hit_rate", "p2_hit_rate",
        "mean_target_step_s", "mean_target_verify_s")
    for row in rows:
        if any(row.get(key) != value for key, value in common.items()):
            raise RuntimeError(f"seed {seed}: config mismatch at {row.get('uid')}")
        if any(number(row, key) is None for key in required):
            raise RuntimeError(f"seed {seed}: metric missing at {row.get('uid')}")
        for phase in ("p1", "p2"):
            hit = number(row, f"{phase}_hit_rate")
            phase_accept = number(row, f"{phase}_accept_len")
            if hit is not None and hit > 0 and phase_accept is None:
                raise RuntimeError(
                    f"seed {seed}: {phase} AL missing despite a hit at "
                    f"{row.get('uid')}")
        if row.get("error"):
            raise RuntimeError(f"seed {seed}: error at {row.get('uid')}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def fmt_seed(values: list[float]) -> str:
    mean, std = mean_std(values)
    return f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    expected_uids = [row["uid"] for row in load(DATA)]
    results = {}
    for seed in SEEDS:
        path = HERE / f"duet_tree_rerank_s{seed}_o1024.jsonl"
        if not path.exists():
            continue
        rows = load(path)
        if len(rows) != 560:
            continue
        validate(rows, expected_uids, seed)
        results[seed] = aggregate(rows)

    fields = [
        "seed", "questions", "turns", "verify_steps", "completion_tokens",
        *QUESTION_METRICS, "target_step_ms", "target_verify_ms",
        "outside_verify_ms"]
    with (HERE / "overall.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed, result in results.items():
            writer.writerow({"seed": seed, **{
                key: result.get(key) for key in fields[1:]}})

    lines = [
        "# DUET tree with fused P1 rerank: full Spec-Bench, three seeds",
        "",
        "- Full Spec-Bench: 480 questions / 560 turns per seed",
        "- Seeds: 1, 42, 123; output 1,024; profiler off",
        "- MT-Bench two turns are merged before prompt-level averaging",
        "- Tree policy is unchanged: K1/K2=8/4, N1/M1=14/12, N2/M2=8/8",
        "- `SSD_P1_RERANK_PRECOMPUTE=1` and GPU target topology enabled",
        "",
        "## Completion",
        "",
    ]
    for seed in SEEDS:
        path = HERE / f"duet_tree_rerank_s{seed}_o1024.jsonl"
        count = sum(1 for line in path.open() if line.strip()) \
            if path.exists() else 0
        lines.append(f"- seed {seed}: {count}/560")

    lines += [
        "", "## Overall (prompt-level)", "",
        "| Seed | Questions (turns) | Decode TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step ms | Target verify ms | Outside verify ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        if seed not in results:
            continue
        r = results[seed]
        lines.append(
            f"| {seed} | {r['questions']} ({r['turns']}) | {fmt(r['tps'])} | "
            f"{fmt(r['accept_len'])} | {fmt(r['cache_hit_rate'])} | "
            f"{fmt(r['p1_hit_rate'])} | {fmt(r['p1_accept_len'])} | "
            f"{fmt(r['p2_hit_rate'])} | {fmt(r['p2_accept_len'])} | "
            f"{fmt(r['target_step_ms'])} | {fmt(r['target_verify_ms'])} | "
            f"{fmt(r['outside_verify_ms'])} |")
    if len(results) == 3:
        metric_map = (
            "tps", "accept_len", "cache_hit_rate", "p1_hit_rate",
            "p1_accept_len", "p2_hit_rate", "p2_accept_len",
            "target_step_ms", "target_verify_ms", "outside_verify_ms")
        values = {
            key: [results[seed][key] for seed in SEEDS]
            for key in metric_map}
        lines.append(
            "| Mean ± SD | 480 (560) | "
            + " | ".join(fmt_seed(values[key]) for key in metric_map)
            + " |")

    for metric, title in (("tps", "Decode TPS"),
                          ("accept_len", "Accepted length"),
                          ("cache_hit_rate", "Cache hit rate"),
                          ("p1_accept_len", "P1 conditional AL"),
                          ("p2_accept_len", "P2 conditional AL")):
        lines += [
            "", f"## {title} by subtask", "",
            "| Seed | " + " | ".join(GROUPS) + " | Overall |",
            "|---:|" + "---:|" * 7,
        ]
        for seed in SEEDS:
            if seed not in results:
                continue
            r = results[seed]
            group_values = [
                fmt(r["groups"][group][metric]) for group in GROUPS]
            lines.append(
                f"| {seed} | " + " | ".join(group_values)
                + f" | {fmt(r[metric])} |")
        if len(results) == 3:
            group_cells = []
            for group in GROUPS:
                group_cells.append(fmt_seed([
                    results[seed]["groups"][group][metric]
                    for seed in SEEDS]))
            lines.append(
                "| Mean ± SD | " + " | ".join(group_cells) + " | "
                + fmt_seed([results[seed][metric] for seed in SEEDS]) + " |")

    lines += [
        "", "## Integrity", "",
        f"- Complete validated seeds: {len(results)}/3",
        "- Each completed seed must match all 560 dataset UIDs in exact order",
        "- Each row is checked for the fixed tree configuration and metrics",
        "- TPS is decode-only and aggregated at prompt/question level",
        "", f"Machine-readable table: `{HERE / 'overall.csv'}`",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(HERE / "RESULTS.md")


if __name__ == "__main__":
    main()
