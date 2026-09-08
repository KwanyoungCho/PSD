#!/usr/bin/env python3
"""Create a fixed balanced 120-question phase-difficulty subset."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path("/home/eslab/chokwans99/baseline")
SRC = BASE / "data/specbench_full.jsonl"
OUT = Path(__file__).resolve().parent / "balanced_120q.jsonl"
GROUPS = ("mt_bench", "translation", "summarization", "qa",
          "math_reasoning", "rag")
SEED = 20260814

rows = [json.loads(line) for line in SRC.open() if line.strip()]
by_group_qid: dict[str, dict[object, list[dict]]] = defaultdict(
    lambda: defaultdict(list))
for row in rows:
    by_group_qid[row["group"]][row["question_id"]].append(row)

rng = random.Random(SEED)
selected: set[tuple[str, object]] = set()
for group in GROUPS:
    qids = sorted(by_group_qid[group])
    picked = rng.sample(qids, 20)
    selected.update((group, qid) for qid in picked)

chosen = [row for row in rows
          if (row["group"], row["question_id"]) in selected]
with OUT.open("w") as handle:
    for row in chosen:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"wrote {OUT}: {len(selected)} questions / {len(chosen)} turns")
for group in GROUPS:
    group_rows = [r for r in chosen if r["group"] == group]
    print(group, len({r["question_id"] for r in group_rows}), len(group_rows))
