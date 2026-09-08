#!/usr/bin/env python3
"""Create a fixed 30-question tuning / 90-question held-out split."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "balanced_120q_tree_safe.jsonl"
TUNE = HERE / "calibration_tune_30q.jsonl"
HELDOUT = HERE / "calibration_heldout_90q.jsonl"
SEED = 20260814

rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
questions = defaultdict(list)
for row in rows:
    questions[(row["group"], row["question_id"])].append(row)

rng = random.Random(SEED)
tune_keys = set()
for group in sorted({group for group, _ in questions}):
    group_keys = sorted(key for key in questions if key[0] == group)
    tune_keys.update(rng.sample(group_keys, 5))

for path, predicate in (
        (TUNE, lambda key: key in tune_keys),
        (HELDOUT, lambda key: key not in tune_keys)):
    selected = [row for row in rows
                if predicate((row["group"], row["question_id"]))]
    with path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path.name, len({(r['group'], r['question_id']) for r in selected}),
          "questions", len(selected), "turns")
