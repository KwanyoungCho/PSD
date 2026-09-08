#!/usr/bin/env python3
"""Validate and compare the full seed-42 matched chain/tree DUET runs."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/eslab/chokwans99/PSD/ssd")
BASE = Path("/home/eslab/chokwans99/baseline")
HERE = ROOT / (
    "experiments/proxy_async_overlap/tree_sweep/"
    "p1_p2_tree_matched_chain_seed42_20260813")
CHAIN = HERE / "duet_chain_matched_treecfg_s42_o1024.jsonl"
TREE = ROOT / (
    "experiments/proxy_async_overlap/tree_sweep/"
    "p1_p2_tree_full_rerank_3seed_20260812/"
    "duet_tree_rerank_s42_o1024.jsonl")
DATA = BASE / "data/specbench_full.jsonl"
GROUPS = ("mt_bench", "translation", "summarization", "qa",
          "math_reasoning", "rag")
METRICS = ("tps", "accept_len", "cache_hit_rate", "p1_hit_rate",
           "p1_accept_len", "p2_hit_rate", "p2_accept_len")


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def questions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["question_id"])].append(row)
    result = []
    for (group, question_id), turns in grouped.items():
        turns.sort(key=lambda row: row.get("turn", 0))
        tokens = sum(number(row, "decode_total_tokens") or 0 for row in turns)
        elapsed = sum(number(row, "decode_total_time") or 0 for row in turns)
        result.append({
            "key": (group, question_id),
            "group": group,
            "turns": len(turns),
            "tps": tokens / elapsed,
            "accept_len": weighted(turns, "accept_len"),
            "cache_hit_rate": weighted(turns, "cache_hit_rate"),
            "p1_hit_rate": weighted(turns, "p1_hit_rate"),
            "p1_accept_len": phase_al(turns, "p1"),
            "p2_hit_rate": weighted(turns, "p2_hit_rate"),
            "p2_accept_len": phase_al(turns, "p2"),
        })
    return result


def mean_present(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows
              if isinstance(row.get(key), (int, float))]
    return statistics.fmean(values) if values else None


def step_ms(rows: list[dict], key: str) -> float | None:
    value = weighted(rows, key)
    return None if value is None else 1000.0 * value


def aggregate(rows: list[dict]) -> dict:
    prompts = questions(rows)
    result = {
        "questions": len(prompts),
        "turns": len(rows),
        **{key: mean_present(prompts, key) for key in METRICS},
        "target_step_ms": step_ms(rows, "mean_target_step_s"),
        "target_verify_ms": step_ms(rows, "mean_target_verify_s"),
    }
    result["outside_verify_ms"] = (
        result["target_step_ms"] - result["target_verify_ms"])
    result["groups"] = {}
    for group in GROUPS:
        selected = [row for row in prompts if row["group"] == group]
        result["groups"][group] = {
            "questions": len(selected),
            "turns": sum(row["turns"] for row in selected),
            **{key: mean_present(selected, key) for key in METRICS},
        }
    return result


def validate(rows: list[dict], expected_uids: list[str], topology: str) -> None:
    if len(rows) != 560:
        raise RuntimeError(f"{topology}: incomplete ({len(rows)}/560)")
    if [row.get("uid") for row in rows] != expected_uids:
        raise RuntimeError(f"{topology}: UID/order mismatch")
    common = {
        "engine": "duet-current(flashinfer,graph)",
        "target": "facebook/layerskip-llama2-70B",
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpus": 3, "k1": 8, "k2": 4, "exit_layer": 56,
        "p1_fanout": 3, "p2_budget": 15, "proxy_top_k": 28,
        "effective_proxy_top_k": 28,
        "p1_allocation_policy": "backbone",
        "c_tensor": 2, "n1": 14, "n2": 8,
        "p1_verify_nodes": 12, "p2_verify_nodes": 8,
        "temp": 0.7, "top_p": 1.0,
        "max_new_tokens": 1024, "max_model_len": 4096,
        "extend_draft_rope": True, "seed": 42, "sampler_seed": 42,
        "p1_tree": topology, "p2_tree": topology,
    }
    required = (
        "decode_total_tokens", "decode_total_time", "accept_len",
        "cache_hit_rate", "p1_hit_rate", "p2_hit_rate",
        "mean_target_step_s", "mean_target_verify_s", "n_verify_steps")
    for row in rows:
        mismatched = [key for key, value in common.items()
                      if row.get(key) != value]
        if mismatched:
            raise RuntimeError(
                f"{topology}: config mismatch at {row.get('uid')}: "
                + ", ".join(mismatched))
        if any(number(row, key) is None for key in required):
            raise RuntimeError(f"{topology}: metric missing at {row.get('uid')}")
        if row.get("error"):
            raise RuntimeError(f"{topology}: request error at {row.get('uid')}")


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def delta(chain: float | None, tree: float | None) -> str:
    if chain is None or tree is None:
        return "—"
    return f"{tree - chain:+.3f}"


def main() -> None:
    expected_uids = [row["uid"] for row in load(DATA)]
    chain_rows, tree_rows = load(CHAIN), load(TREE)
    validate(chain_rows, expected_uids, "off")
    validate(tree_rows, expected_uids, "on")
    results = {
        "DUET-chain (matched)": aggregate(chain_rows),
        "DUET-tree": aggregate(tree_rows),
    }

    fields = ["method", "questions", "turns", *METRICS,
              "target_step_ms", "target_verify_ms", "outside_verify_ms"]
    with (HERE / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in results.items():
            writer.writerow({"method": method, **{
                key: result.get(key) for key in fields[1:]}})

    chain, tree = results.values()
    lines = [
        "# DUET matched chain vs. tree (full Spec-Bench, seed 42)", "",
        "- Both arms use the latest DUET engine and the same 560 turns.",
        "- Fixed: seed 42, output 1,024, K1/K2=8/4, exit layer 56, "
        "proxy top-k 28, P1 fanout 3, P2 budget 15, N1/M1=14/12, "
        "N2/M2=8/8.",
        "- Changed: only the candidate topology policy, chain "
        "(`P1/P2 tree=off`) vs. tree (`P1/P2 tree=on`).",
        "- Decode TPS and AL/hit metrics are question-level means; MT-Bench "
        "turns are merged before averaging. Step latency is weighted by "
        "verification-step count.", "",
        "## Overall", "",
        "| Method | Questions (turns) | Decode TPS | AL | Hit | P1 hit | "
        "P1 AL | P2 hit | P2 AL | Target step ms | Target verify ms | "
        "Outside verify ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in results.items():
        lines.append(
            f"| {method} | {result['questions']} ({result['turns']}) | "
            f"{fmt(result['tps'])} | {fmt(result['accept_len'])} | "
            f"{fmt(result['cache_hit_rate'])} | "
            f"{fmt(result['p1_hit_rate'])} | "
            f"{fmt(result['p1_accept_len'])} | "
            f"{fmt(result['p2_hit_rate'])} | "
            f"{fmt(result['p2_accept_len'])} | "
            f"{fmt(result['target_step_ms'])} | "
            f"{fmt(result['target_verify_ms'])} | "
            f"{fmt(result['outside_verify_ms'])} |")
    lines.append(
        "| Tree − chain | — | "
        + " | ".join(delta(chain[key], tree[key]) for key in (
            "tps", "accept_len", "cache_hit_rate", "p1_hit_rate",
            "p1_accept_len", "p2_hit_rate", "p2_accept_len",
            "target_step_ms", "target_verify_ms", "outside_verify_ms"))
        + " |")

    for metric, title in (("tps", "Decode TPS"),
                          ("accept_len", "Accepted length")):
        lines += ["", f"## {title} by subtask", "",
                  "| Method | " + " | ".join(GROUPS) + " | Overall |",
                  "|---|" + "---:|" * 7]
        for method, result in results.items():
            cells = [fmt(result["groups"][group][metric])
                     for group in GROUPS]
            lines.append(
                f"| {method} | " + " | ".join(cells)
                + f" | {fmt(result[metric])} |")

    lines += ["", "## Integrity", "",
              "- Both files validated as exactly 480 questions / 560 turns.",
              "- Dataset UID and order must match exactly.",
              "- All fixed configuration fields and required metrics are "
              "validated row by row.", "",
              f"Machine-readable table: `{HERE / 'comparison.csv'}`"]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(HERE / "RESULTS.md")


if __name__ == "__main__":
    main()
