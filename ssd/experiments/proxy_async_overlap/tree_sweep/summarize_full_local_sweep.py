#!/usr/bin/env python3
"""Validate and summarize the full-data DUET-tree local sweep."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path("/home/eslab/chokwans99/PSD/ssd")
BASE = Path("/home/eslab/chokwans99/baseline")
HERE = ROOT / "experiments/proxy_async_overlap/tree_sweep/p1_p2_tree_full_local_sweep_seed42_20260812"
DATA = BASE / "data/specbench_full.jsonl"
ORIGINAL = ROOT / (
    "experiments/proxy_async_overlap/tree_sweep/"
    "p1_p2_tree_full_specbench_seed42_20260811/p1_backbone_s42_o1024.jsonl"
)
GROUPS = ("mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag")
CASES = {
    "original_seed42": (ORIGINAL, 2, 14, 12, 0.0, 0.0),
    "reference_repeat": (HERE / "reference_repeat_s42_o1024.jsonl", 2, 14, 12, 0.0, 0.0),
    "n1_12": (HERE / "n1_12_s42_o1024.jsonl", 2, 12, 12, 0.0, 0.0),
    "c3": (HERE / "c3_s42_o1024.jsonl", 3, 14, 12, 0.0, 0.0),
    "threshold_mild": (HERE / "threshold_mild_s42_o1024.jsonl", 2, 14, 12, 0.001, 0.01),
}
METRICS = (
    "tps", "accept_len", "cache_hit_rate", "p1_hit_rate", "p1_accept_len",
    "p2_hit_rate", "p2_accept_len", "target_step_ms", "target_verify_ms",
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def weighted(rows: list[dict], key: str, weight_key: str = "n_verify_steps") -> float | None:
    pairs = []
    for row in rows:
        value, weight = number(row, key), number(row, weight_key)
        if value is not None and weight is not None and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return None
    return sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)


def phase_al(rows: list[dict], phase: str) -> float | None:
    pairs = []
    for row in rows:
        value = number(row, f"{phase}_accept_len")
        steps = number(row, "n_verify_steps")
        hit = number(row, f"{phase}_hit_rate")
        if value is not None and steps is not None and hit is not None and steps * hit > 0:
            pairs.append((value, steps * hit))
    if not pairs:
        return None
    return sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)


def make_questions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["question_id"])].append(row)
    result = []
    for (group, qid), turns in grouped.items():
        turns.sort(key=lambda row: row.get("turn", 0))
        tokens = sum(number(row, "decode_total_tokens") or 0 for row in turns)
        elapsed = sum(number(row, "decode_total_time") or 0 for row in turns)
        result.append({
            "key": (group, qid),
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
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return fmean(values) if values else None


def step_ms(rows: list[dict], key: str) -> float | None:
    pairs = []
    for row in rows:
        value, steps = number(row, key), number(row, "n_verify_steps")
        if value is not None and steps is not None and steps > 0:
            pairs.append((value, steps))
    if not pairs:
        return None
    return 1000 * sum(value * steps for value, steps in pairs) / sum(steps for _, steps in pairs)


def aggregate(rows: list[dict]) -> dict:
    questions = make_questions(rows)
    result = {
        "questions": len(questions),
        "turns": len(rows),
        **{key: mean_present(questions, key) for key in METRICS[:7]},
        "target_step_ms": step_ms(rows, "mean_target_step_s"),
        "target_verify_ms": step_ms(rows, "mean_target_verify_s"),
    }
    result["groups"] = {}
    for group in GROUPS:
        selected = [row for row in questions if row["group"] == group]
        result["groups"][group] = {
            "questions": len(selected),
            "turns": sum(row["turns"] for row in selected),
            **{key: mean_present(selected, key) for key in METRICS[:7]},
        }
    return result


def validate(rows: list[dict], expected_uids: list[str], spec: tuple) -> None:
    _, c, n1, m1, start_thr, conf_thr = spec
    if len(rows) != 560 or [row.get("uid") for row in rows] != expected_uids:
        raise RuntimeError("result does not exactly match the 560-turn Spec-Bench input")
    required = (
        "decode_tps", "accept_len", "cache_hit_rate", "p1_hit_rate",
        "p2_hit_rate", "mean_target_step_s", "mean_target_verify_s",
    )
    common = {
        "engine": "duet-current(flashinfer,graph)", "k1": 8, "k2": 4,
        "exit_layer": 56, "p1_fanout": 3, "p2_budget": 15, "proxy_top_k": 28,
        "p1_tree": "on", "p2_tree": "on", "p1_allocation_policy": "backbone",
        "n2": 8, "p1_verify_nodes": m1, "p2_verify_nodes": 8,
        "max_new_tokens": 1024, "max_model_len": 4096,
        "extend_draft_rope": True, "seed": 42, "sampler_seed": 42,
        "c_tensor": c, "n1": n1,
    }
    for row in rows:
        if any(row.get(key) != value for key, value in common.items()):
            raise RuntimeError(f"unexpected config in {row.get('uid')}")
        if any(row.get(key) is None for key in required):
            raise RuntimeError(f"missing metric in {row.get('uid')}")
    # Thresholds are not serialized by the runner, so they are validated by the
    # immutable command table in the harness and recorded in this report.
    assert start_thr >= 0 and conf_thr >= 0


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    expected_uids = [row["uid"] for row in load(DATA)]
    completed: dict[str, dict] = {}
    raw_rows: dict[str, list[dict]] = {}
    for name, spec in CASES.items():
        path = spec[0]
        if not path.exists() or sum(1 for line in path.open() if line.strip()) != 560:
            continue
        rows = load(path)
        validate(rows, expected_uids, spec)
        raw_rows[name] = rows
        completed[name] = aggregate(rows)

    overall_csv = HERE / "overall.csv"
    with overall_csv.open("w", newline="") as handle:
        fields = ["case", "c", "n1", "m1", "p1_start_threshold", "p1_conf_threshold", "questions", "turns", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in completed.items():
            _, c, n1, m1, start_thr, conf_thr = CASES[name]
            writer.writerow({"case": name, "c": c, "n1": n1, "m1": m1,
                             "p1_start_threshold": start_thr, "p1_conf_threshold": conf_thr,
                             **{key: result.get(key) for key in fields[6:]}})

    lines = [
        "# DUET P1+P2 tree full Spec-Bench local sweep",
        "",
        "모든 arm은 전체 Spec-Bench 480 questions / 560 turns를 사용한다. MT-Bench의",
        "두 turn은 먼저 한 question으로 결합하며, TPS는 decode-only tokens/time을",
        "question마다 계산한 뒤 평균한다. 모든 arm은 seed 42, output 1,024다.",
        "",
        "## Sweep design",
        "",
        "| Case | C | N1 | M1 | P1 start/conf threshold | Changed from reference |",
        "|---|---:|---:|---:|---:|---|",
        "| original_seed42 | 2 | 14 | 12 | 0 / 0 | 2026-08-11 reference raw |",
        "| reference_repeat | 2 | 14 | 12 | 0 / 0 | exact independent repeat |",
        "| n1_12 | 2 | 12 | 12 | 0 / 0 | generated nodes/root 14→12; removes on-hit rerank |",
        "| c3 | 3 | 14 | 12 | 0 / 0 | branch width 2→3 only |",
        "| threshold_mild | 2 | 14 | 12 | 0.001 / 0.01 | mild P1 pruning only |",
        "",
        "`M1=12`와 P2 설정은 전 arm에서 고정한다. 따라서 C/N1 비교는 target에 보내는",
        "P1 verification node cap을 바꾸지 않는다. 공통 P2는 budget/root/N2/M2",
        "=15/10/8/8, threshold=0.01/0.01이다.",
        "실행은 target-step 증가 원인 분석을 위해 일시 중단했다. `N1=12` arm은",
        "분석에서 확인한 on-hit 14→12 rerank 비용을 없애면서 M1을 고정하는 진단 arm이다.",
        "",
        "## Completion status",
        "",
    ]
    for name, spec in CASES.items():
        path = spec[0]
        rows = sum(1 for line in path.open() if line.strip()) if path.exists() else 0
        lines.append(f"- `{name}`: {rows}/560 turns")

    lines += [
        "",
        "## Overall",
        "",
        "| Case | Questions (turns) | Decode TPS | AL | P1 AL | P2 AL | Hit | P1 hit | P2 hit | Target step ms | Target verify ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in completed.items():
        lines.append(
            f"| {name} | {result['questions']} ({result['turns']}) | {fmt(result['tps'])} | "
            f"{fmt(result['accept_len'])} | {fmt(result['p1_accept_len'])} | {fmt(result['p2_accept_len'])} | "
            f"{fmt(result['cache_hit_rate'])} | {fmt(result['p1_hit_rate'])} | {fmt(result['p2_hit_rate'])} | "
            f"{fmt(result['target_step_ms'])} | {fmt(result['target_verify_ms'])} |"
        )

    if "reference_repeat" in completed:
        ref = completed["reference_repeat"]
        lines += ["", "## Delta from this sweep's reference repeat", "",
                  "| Case | Δ TPS | Δ AL | Δ P1 AL | Δ P2 AL | Δ target step ms |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, result in completed.items():
            if name == "reference_repeat":
                continue
            lines.append(
                f"| {name} | {result['tps']-ref['tps']:+.3f} | "
                f"{result['accept_len']-ref['accept_len']:+.3f} | "
                f"{result['p1_accept_len']-ref['p1_accept_len']:+.3f} | "
                f"{result['p2_accept_len']-ref['p2_accept_len']:+.3f} | "
                f"{result['target_step_ms']-ref['target_step_ms']:+.3f} |"
            )

    for metric, title in (("tps", "Decode TPS"), ("accept_len", "Accepted length"),
                          ("p1_accept_len", "P1 conditional AL")):
        lines += ["", f"## {title} by subtask", "",
                  "| Case | " + " | ".join(GROUPS) + " | Overall |",
                  "|---|" + "---:|" * 7]
        for name, result in completed.items():
            values = [fmt(result["groups"][group][metric]) for group in GROUPS]
            lines.append(f"| {name} | " + " | ".join(values) + f" | {fmt(result[metric])} |")

    if "original_seed42" in raw_rows and "reference_repeat" in raw_rows:
        old = {row["uid"]: row for row in raw_rows["original_seed42"]}
        new = {row["uid"]: row for row in raw_rows["reference_repeat"]}
        hash_matches = sum(old[uid].get("output_sha256") == new[uid].get("output_sha256") for uid in old)
        lines += [
            "", "## Repeatability note", "",
            f"- Exact output hash matches: {hash_matches}/560 turns.",
            "- 같은 seed라도 GPU sampling/execution은 bitwise deterministic하다고 가정하지 않는다.",
            "- parameter arm의 판단 기준은 이번 `reference_repeat`이며, 기존 원본과의 차이는",
            "  독립 재실행 편차를 보여주는 보조 비교로만 사용한다.",
        ]

    lines += [
        "", "## Fixed configuration", "",
        "- Target/draft: LayerSkip Llama-2-70B / TinyLlama-1.1B",
        "- K1/K2=8/4, exit layer 56, P1 fan-out/roots-per-position=3/3",
        "- proxy top-k 28; P1 allocation `backbone`; P1/P2 tree on",
        "- P2 budget/root/N2/M2=15/10/8/8; P2 thresholds 0.01/0.01",
        "- raw prompt, temperature 0.7, top-p 1.0, context 4,096, draft RoPE extension",
        "", f"Machine-readable overall table: `{overall_csv}`",
    ]
    (HERE / "SWEEP_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(HERE / "SWEEP_RESULTS.md")


if __name__ == "__main__":
    main()
