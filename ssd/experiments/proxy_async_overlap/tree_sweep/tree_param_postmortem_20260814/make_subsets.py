#!/usr/bin/env python3
"""Create fixed forensic and screening question subsets for tree tuning."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
DATA = Path("/home/eslab/chokwans99/baseline/data/specbench_full.jsonl")
REFERENCE = Path(
    "/home/eslab/chokwans99/PSD/ssd/experiments/proxy_async_overlap/"
    "tree_sweep/p1_p2_tree_full_rerank_3seed_20260812/"
    "duet_tree_rerank_s42_o1024.jsonl"
)
GROUPS = (
    "mt_bench", "translation", "summarization", "qa",
    "math_reasoning", "rag",
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def weighted(rows: list[dict], key: str, weight_key: str = "n_verify_steps"):
    pairs = [(float(r[key]), float(r[weight_key])) for r in rows
             if isinstance(r.get(key), (int, float))
             and isinstance(r.get(weight_key), (int, float))
             and float(r[weight_key]) > 0]
    return sum(x * w for x, w in pairs) / sum(w for _, w in pairs) \
        if pairs else math.nan


def question_features(rows: list[dict]) -> dict:
    steps = sum(float(r.get("n_verify_steps") or 0) for r in rows)
    p1_steps = sum(float(r.get("n_verify_steps") or 0) *
                   float(r.get("p1_hit_rate") or 0) for r in rows)
    p2_steps = sum(float(r.get("n_verify_steps") or 0) *
                   float(r.get("p2_hit_rate") or 0) for r in rows)

    def phase_al(phase: str, phase_steps: float):
        if phase_steps <= 0:
            return math.nan
        total = 0.0
        for row in rows:
            hit_steps = float(row.get("n_verify_steps") or 0) * \
                float(row.get(f"{phase}_hit_rate") or 0)
            value = row.get(f"{phase}_accept_len")
            if isinstance(value, (int, float)):
                total += hit_steps * float(value)
        return total / phase_steps

    hit = weighted(rows, "cache_hit_rate")
    return {
        "group": rows[0]["group"],
        "question_id": rows[0]["question_id"],
        "turns": len(rows),
        "prompt_tokens_max": max(float(r.get("prefill_total_tokens") or 0)
                                 for r in rows),
        "accept_len": weighted(rows, "accept_len"),
        "hit": hit,
        "p1_hit": p1_steps / steps if steps else 0.0,
        "p2_hit": p2_steps / steps if steps else 0.0,
        "miss": 1.0 - hit if math.isfinite(hit) else math.nan,
        "p1_al_spec_only": phase_al("p1", p1_steps),
        "p2_al_spec_only": phase_al("p2", p2_steps),
        "uids": ",".join(r["uid"] for r in sorted(
            rows, key=lambda x: x.get("turn", 0))),
    }


def first_unused(rows: list[dict], selected: set, key, reverse=False,
                 predicate=lambda _: True):
    candidates = sorted((r for r in rows if predicate(r)),
                        key=key, reverse=reverse)
    return next((r for r in candidates if r["question_id"] not in selected), None)


def forensic(rows: list[dict], n: int = 10) -> tuple[list[dict], dict]:
    selected: set = set()
    reasons: dict = defaultdict(list)
    specs = [
        ("long_prompt", lambda r: r["prompt_tokens_max"], True,
         lambda r: True),
        ("low_total_al", lambda r: r["accept_len"], False,
         lambda r: True),
        ("high_total_al", lambda r: r["accept_len"], True,
         lambda r: True),
        ("p1_heavy", lambda r: r["p1_hit"], True, lambda r: True),
        ("p2_heavy", lambda r: r["p2_hit"], True, lambda r: True),
        ("miss_heavy", lambda r: r["miss"], True, lambda r: True),
        ("low_p1_al", lambda r: r["p1_al_spec_only"], False,
         lambda r: math.isfinite(r["p1_al_spec_only"])),
        ("high_p1_al", lambda r: r["p1_al_spec_only"], True,
         lambda r: math.isfinite(r["p1_al_spec_only"])),
        ("high_p2_al", lambda r: r["p2_al_spec_only"], True,
         lambda r: math.isfinite(r["p2_al_spec_only"])),
    ]
    for reason, key, reverse, predicate in specs:
        row = first_unused(rows, selected, key, reverse, predicate)
        if row:
            selected.add(row["question_id"])
            reasons[row["question_id"]].append(reason)

    # Add a central example, then maximin-fill in rank-normalized behavior
    # space if an extreme criterion collided with an earlier selection.
    fields = ("prompt_tokens_max", "accept_len", "p1_hit", "p2_hit", "miss")
    centers = {k: median(r[k] for r in rows) for k in fields}
    scales = {k: max(max(abs(r[k] - centers[k]) for r in rows), 1e-9)
              for k in fields}

    def vec(row):
        return tuple((row[k] - centers[k]) / scales[k] for k in fields)

    def distance(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(vec(a), vec(b))))

    central = min((r for r in rows if r["question_id"] not in selected),
                  key=lambda r: math.sqrt(sum(x * x for x in vec(r))))
    selected.add(central["question_id"])
    reasons[central["question_id"]].append("median_behavior")
    while len(selected) < n:
        chosen = [r for r in rows if r["question_id"] in selected]
        row = max((r for r in rows if r["question_id"] not in selected),
                  key=lambda r: min(distance(r, x) for x in chosen))
        selected.add(row["question_id"])
        reasons[row["question_id"]].append("maximin_fill")
    return [r for r in rows if r["question_id"] in selected], reasons


def main() -> None:
    data = load(DATA)
    reference = load(REFERENCE)
    by_q = defaultdict(list)
    for row in reference:
        by_q[(row["group"], row["question_id"])].append(row)
    features = [question_features(rows) for rows in by_q.values()]

    forensic_rows, manifest = [], []
    forensic_ids = set()
    for group in GROUPS:
        one = [r for r in features if r["group"] == group]
        chosen, reasons = forensic(one)
        for row in chosen:
            key = (group, row["question_id"])
            forensic_ids.add(key)
            forensic_rows.append(key)
            manifest.append({"subset": "forensic", "reason": ";".join(
                reasons[row["question_id"]]), **row})

    rng = random.Random(20260814)
    screening_rows = []
    for group in GROUPS:
        available = [r for r in features
                     if r["group"] == group and
                     (group, r["question_id"]) not in forensic_ids]
        chosen = rng.sample(available, 12)
        for row in chosen:
            key = (group, row["question_id"])
            screening_rows.append(key)
            manifest.append({"subset": "screening", "reason": "fixed_random",
                             **row})

    order = {(r["group"], r["question_id"], r["turn"]): i
             for i, r in enumerate(data)}

    def write_subset(name: str, keys: list[tuple]):
        chosen = set(keys)
        rows = [r for r in data if (r["group"], r["question_id"]) in chosen]
        rows.sort(key=lambda r: order[(r["group"], r["question_id"], r["turn"])])
        with (HERE / f"{name}_subset.jsonl").open("w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, len(chosen), "questions", len(rows), "turns")

    write_subset("forensic", forensic_rows)
    write_subset("screening", screening_rows)
    with (HERE / "subset_manifest.csv").open("w", newline="") as out:
        fieldnames = ("subset", "group", "question_id", "turns", "reason",
                      "uids", "prompt_tokens_max", "accept_len", "hit",
                      "p1_hit", "p2_hit", "miss", "p1_al_spec_only",
                      "p2_al_spec_only")
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(manifest, key=lambda r: (
                r["subset"], GROUPS.index(r["group"]), str(r["question_id"]))):
            writer.writerow({k: row.get(k) for k in fieldnames})


if __name__ == "__main__":
    main()
