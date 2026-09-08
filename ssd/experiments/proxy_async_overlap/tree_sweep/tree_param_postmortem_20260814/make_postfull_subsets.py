#!/usr/bin/env python3
"""Create disjoint post-full diagnostic and screening question subsets."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = Path("/home/eslab/chokwans99/baseline/data/specbench_full.jsonl")
REF = HERE / "full/ref_k8_k4_e56_n8m8/raw.jsonl"
WINNER = HERE / "full/winner_k8_k5_e49_n10m10/raw.jsonl"
OLD_MANIFEST = HERE / "subset_manifest.csv"
GROUPS = (
    "mt_bench", "translation", "summarization", "qa",
    "math_reasoning", "rag",
)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def grouped(rows: list[dict]) -> dict[tuple[str, object], list[dict]]:
    out: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for row in rows:
        out[(row["group"], row["question_id"])].append(row)
    return out


def weighted(rows: list[dict], key: str, phase: str | None = None) -> float:
    pairs = []
    for row in rows:
        value = row.get(key)
        steps = row.get("n_verify_steps")
        if not isinstance(value, (int, float)) or not isinstance(steps, (int, float)):
            continue
        weight = float(steps)
        if phase is not None:
            hit = row.get(f"{phase}_hit_rate")
            if not isinstance(hit, (int, float)):
                continue
            weight *= float(hit)
        if weight > 0:
            pairs.append((float(value), weight))
    if not pairs:
        return math.nan
    return sum(value * weight for value, weight in pairs) / sum(
        weight for _, weight in pairs
    )


def features(rows: list[dict]) -> dict:
    return {
        "group": rows[0]["group"],
        "question_id": rows[0]["question_id"],
        "turns": len(rows),
        "prompt_tokens_max": max(float(row.get("prompt_tokens") or 0) for row in rows),
        "verify_steps": sum(float(row.get("n_verify_steps") or 0) for row in rows),
        "accept_len": weighted(rows, "accept_len"),
        "hit": weighted(rows, "cache_hit_rate"),
        "p1_hit": weighted(rows, "p1_hit_rate"),
        "p2_hit": weighted(rows, "p2_hit_rate"),
        "p1_al": weighted(rows, "p1_accept_len", "p1"),
        "p2_al": weighted(rows, "p2_accept_len", "p2"),
        "uids": ",".join(row["uid"] for row in sorted(
            rows, key=lambda row: row.get("turn", 0))),
    }


def finite(row: dict, key: str) -> bool:
    return isinstance(row.get(key), (int, float)) and math.isfinite(row[key])


def choose_diagnostic(rows: list[dict], count: int = 8):
    selected: set[object] = set()
    reasons: dict[object, list[str]] = defaultdict(list)
    rules = (
        ("p2_heavy", "p2_hit", True),
        ("low_p2_al", "p2_al", False),
        ("high_p2_al", "p2_al", True),
        ("largest_p2_hit_drop", "p2_hit_delta", False),
        ("smallest_al_gain", "accept_len_delta", False),
        ("largest_al_gain", "accept_len_delta", True),
        ("long_prompt", "prompt_tokens_max", True),
        ("many_verify_steps", "verify_steps", True),
    )
    for reason, key, reverse in rules:
        candidates = [row for row in rows if finite(row, key)]
        candidates.sort(key=lambda row: row[key], reverse=reverse)
        picked = next((row for row in candidates
                       if row["question_id"] not in selected), None)
        if picked is not None:
            selected.add(picked["question_id"])
            reasons[picked["question_id"]].append(reason)

    # Collision-safe deterministic fill by total behavior extremes.
    while len(selected) < count:
        remaining = [row for row in rows if row["question_id"] not in selected]
        picked = max(remaining, key=lambda row: (
            abs(row["accept_len_delta"]), row["verify_steps"],
            str(row["question_id"])))
        selected.add(picked["question_id"])
        reasons[picked["question_id"]].append("collision_fill")
    return [row for row in rows if row["question_id"] in selected], reasons


def main() -> None:
    data = load(DATA)
    ref = grouped(load(REF))
    winner = grouped(load(WINNER))
    if set(ref) != set(winner):
        raise ValueError("reference and winner question sets differ")

    used: set[tuple[str, str]] = set()
    with OLD_MANIFEST.open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            used.add((row["group"], row["question_id"]))

    rows = []
    for key in sorted(winner, key=lambda key: (GROUPS.index(key[0]), str(key[1]))):
        win = features(winner[key])
        base = features(ref[key])
        win["accept_len_delta"] = win["accept_len"] - base["accept_len"]
        win["p2_hit_delta"] = win["p2_hit"] - base["p2_hit"]
        rows.append(win)

    diagnostic_keys = []
    manifest = []
    for group in GROUPS:
        available = [row for row in rows
                     if row["group"] == group and
                     (group, str(row["question_id"])) not in used]
        chosen, reasons = choose_diagnostic(available)
        for row in chosen:
            key = (group, str(row["question_id"]))
            used.add(key)
            diagnostic_keys.append((group, row["question_id"]))
            manifest.append({"subset": "postfull_forensic",
                             "reason": ";".join(reasons[row["question_id"]]),
                             **row})

    rng = random.Random(20260815)
    screening_keys = []
    for group in GROUPS:
        available = [row for row in rows
                     if row["group"] == group and
                     (group, str(row["question_id"])) not in used]
        chosen = rng.sample(available, 10)
        for row in chosen:
            used.add((group, str(row["question_id"])))
            screening_keys.append((group, row["question_id"]))
            manifest.append({"subset": "postfull_screening",
                             "reason": "fixed_random_seed20260815", **row})

    order = {(row["group"], row["question_id"], row["turn"]): index
             for index, row in enumerate(data)}

    def write_subset(name: str, keys: list[tuple[str, object]]) -> None:
        selected = set(keys)
        out_rows = [row for row in data
                    if (row["group"], row["question_id"]) in selected]
        out_rows.sort(key=lambda row: order[
            (row["group"], row["question_id"], row["turn"])])
        with (HERE / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in out_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, len(selected), "questions", len(out_rows), "turns")

    write_subset("postfull_forensic_subset", diagnostic_keys)
    write_subset("postfull_screening_subset", screening_keys)
    fields = (
        "subset", "group", "question_id", "turns", "reason", "uids",
        "prompt_tokens_max", "verify_steps", "accept_len", "hit", "p1_hit",
        "p2_hit", "p1_al", "p2_al", "accept_len_delta", "p2_hit_delta",
    )
    with (HERE / "postfull_subset_manifest.csv").open(
            "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in sorted(manifest, key=lambda row: (
                row["subset"], GROUPS.index(row["group"]),
                str(row["question_id"]))):
            writer.writerow({key: row.get(key) for key in fields})


if __name__ == "__main__":
    main()
