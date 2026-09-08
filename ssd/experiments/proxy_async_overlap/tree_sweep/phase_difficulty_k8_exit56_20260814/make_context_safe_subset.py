#!/usr/bin/env python3
"""Replace only unsafe questions in the fixed 120-question subset.

The eligibility rule is fixed from the requested generation cap and the
worst-case tree scheduler reservation, not from observed EOS or AL outcomes.
This keeps chain/tree coverage identical without rerunning the 114 questions
that were already safe.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SRC = Path("/home/eslab/chokwans99/baseline/data/specbench_full.jsonl")
ORIGINAL = HERE / "balanced_120q.jsonl"
OUT = HERE / "balanced_120q_tree_safe.jsonl"
TOKENIZER = Path(
    "/home/eslab/models/hub/models--facebook--layerskip-llama2-70B/"
    "snapshots/0a1815fcbd11543b0de227aa8dcbced952149411")

MAX_MODEL_LEN = 2048
MAX_NEW_TOKENS = 512
K1 = K2 = 8
P1_VERIFY_NODES = P2_VERIFY_NODES = 12
ROOTS_PER_POSITION = 3
P2_ROOT_BUDGET = 15
SEED = 20260814

# Mirrors Scheduler for P1 backbone tree / P2 tree.
context_cap = max(K1 + 1, K2 + 1,
                  P1_VERIFY_NODES + 1, P2_VERIFY_NODES + 1)
p1_root_width = context_cap * ROOTS_PER_POSITION
p1_cells = p1_root_width + (K1 - 1) * p1_root_width
p2_cells = K2 * P2_ROOT_BUDGET
glue_width = context_cap
TREE_DRAFT_LOOKAHEAD = glue_width + max(p1_cells, p2_cells)
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS - TREE_DRAFT_LOOKAHEAD

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
rows = [json.loads(line) for line in SRC.open() if line.strip()]
original_rows = [json.loads(line) for line in ORIGINAL.open() if line.strip()]
by_question = defaultdict(list)
for row in rows:
    by_question[(row["group"], row["question_id"])].append(row)


def prompt_len(row):
    return len(tokenizer.encode(row["prompt"]))


question_max_prompt = {
    key: max(prompt_len(row) for row in question_rows)
    for key, question_rows in by_question.items()
}
original_keys = {(row["group"], row["question_id"])
                 for row in original_rows}
unsafe = {key for key in original_keys
          if question_max_prompt[key] > MAX_PROMPT_TOKENS}
selected = original_keys - unsafe

rng = random.Random(SEED + 1)
replacements = []
for group in sorted({group for group, _ in unsafe}):
    needed = sum(key[0] == group for key in unsafe)
    candidates = [key for key in sorted(by_question)
                  if key[0] == group and key not in original_keys
                  and question_max_prompt[key] <= MAX_PROMPT_TOKENS]
    picked = rng.sample(candidates, needed)
    selected.update(picked)
    replacements.extend(picked)

chosen = [row for row in rows
          if (row["group"], row["question_id"]) in selected]
with OUT.open("w") as handle:
    for row in chosen:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"tree draft lookahead: {TREE_DRAFT_LOOKAHEAD}")
print(f"max eligible prompt tokens: {MAX_PROMPT_TOKENS}")
print(f"unsafe original: {sorted(unsafe)}")
print(f"replacements: {sorted(replacements)}")
print(f"wrote {OUT}: {len(selected)} questions / {len(chosen)} turns")
for group in sorted({row["group"] for row in chosen}):
    group_rows = [row for row in chosen if row["group"] == group]
    print(group, len({row["question_id"] for row in group_rows}),
          len(group_rows), max(prompt_len(row) for row in group_rows))
