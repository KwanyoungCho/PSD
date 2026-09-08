#!/usr/bin/env python3
"""Build a post-hoc best-subtask candidate from the common safe subset.

This artifact is diagnostic only.  It deliberately selects the highest-TPS
source after observing each subtask, so it must not be presented as a single
seed or an unbiased headline result.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FULL = HERE / "full"
CURRENT_BEST = Path(
    "/home/eslab/chokwans99/DUET_PAPER_RESULTS/experiments/duet_tree/"
    "best_subtask_selected_plus_3seed/"
    "duet_tree_best_subtask_selected_plus_3seed_o1024_ctx2048_safe.jsonl"
)
OPT42 = FULL / "winner_k8_k5_e49_n10m10/raw_ctx2048_safe.jsonl"
OPT123 = FULL / "winner_k8_k5_e49_n10m10_s123/raw_ctx2048_safe.jsonl"
OUT = HERE / "comparison_best_subtask"

SELECTION = {
    "mt_bench": ("optimized seed 42", OPT42),
    "translation": ("optimized seed 123", OPT123),
    "summarization": ("current best-subtask", CURRENT_BEST),
    "qa": ("current best-subtask", CURRENT_BEST),
    "math_reasoning": ("optimized seed 42", OPT42),
    "rag": ("current best-subtask", CURRENT_BEST),
}
EXPECTED_TURNS = {
    "mt_bench": 160,
    "translation": 80,
    "summarization": 56,
    "qa": 80,
    "math_reasoning": 80,
    "rag": 80,
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    cache = {path: load(path) for _, path in SELECTION.values()}
    selected: list[dict] = []
    manifest: dict[str, dict] = {}

    for group, (label, path) in SELECTION.items():
        rows = [row for row in cache[path] if row["group"] == group]
        if len(rows) != EXPECTED_TURNS[group]:
            raise ValueError(
                f"{group}: expected {EXPECTED_TURNS[group]} turns, got {len(rows)}"
            )
        selected.extend(rows)
        manifest[group] = {
            "source": label,
            "path": str(path),
            "turns": len(rows),
            "questions": len({row["question_id"] for row in rows}),
        }

    if len(selected) != 536:
        raise ValueError(f"expected 536 turns, got {len(selected)}")
    uids = [row["uid"] for row in selected]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate uid in expanded best candidate")
    questions = {(row["group"], row["question_id"]) for row in selected}
    if len(questions) != 456:
        raise ValueError(f"expected 456 questions, got {len(questions)}")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "expanded_best_subtask_candidate.jsonl"
    with output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "selection_manifest.json").open("w", encoding="utf-8") as target:
        json.dump(
            {
                "warning": "post-hoc subtask selection; not a single-seed result",
                "questions": len(questions),
                "turns": len(selected),
                "selection": manifest,
                "output": str(output),
            },
            target,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
