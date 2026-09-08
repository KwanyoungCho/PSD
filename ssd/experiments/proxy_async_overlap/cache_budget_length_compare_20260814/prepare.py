#!/usr/bin/env python3
"""Build exact-budget manifests for K=9 and K=8 cache-hit sweeps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = Path("/home/eslab/chokwans99/baseline/data/specbench_cache_budget_k10.jsonl")
EXPECTED_SHA = "81eee810710dfcf62f4fd585bf66ef443f80dd49e8d27fb45c0ac5a9181aec6f"
BUDGETS = tuple(range(2, 9))
METHODS = ("duet", "only_proxy", "geo", "uniform")
GEO_Q = 0.82


def geometric_fanout(position_count: int, total: int) -> list[int]:
    weights = [GEO_Q ** i for i in range(position_count)]
    raw = [total * weight / sum(weights) for weight in weights]
    result = [math.floor(value) for value in raw]
    order = sorted(
        range(position_count),
        key=lambda i: (raw[i] - result[i], -i),
        reverse=True,
    )
    for index in order[: total - sum(result)]:
        result[index] += 1
    assert len(result) == position_count and sum(result) == total
    return result


def build(k: int, rows: list[dict], dataset_sha: str) -> dict:
    position_count = k + 1
    cells = []
    for budget in BUDGETS:
        total = position_count * budget
        cells.extend([
            {
                "method": "duet", "avg_position_budget": budget,
                "total_cache_roots": total, "logical_k1": k,
                "logical_k2": k, "p1_cache_roots": position_count,
                "p2_cache_roots": total - position_count,
            },
            {
                "method": "only_proxy", "avg_position_budget": budget,
                "total_cache_roots": total, "logical_k1": 0,
                "logical_k2": k, "p1_cache_roots": 0,
                "p2_cache_roots": total,
            },
            {
                "method": "geo", "avg_position_budget": budget,
                "total_cache_roots": total,
                "fan_out_list": geometric_fanout(position_count, total),
            },
            {
                "method": "uniform", "avg_position_budget": budget,
                "total_cache_roots": total,
                "fan_out_list": [budget] * position_count,
            },
        ])
    for cell in cells:
        assert cell["total_cache_roots"] == position_count * cell["avg_position_budget"]
        if "fan_out_list" in cell:
            assert len(cell["fan_out_list"]) == position_count
            assert sum(cell["fan_out_list"]) == cell["total_cache_roots"]
    return {
        "experiment": f"cache_budget_hit_k{k}_seed1",
        "dataset": str(DATASET),
        "dataset_sha256": dataset_sha,
        "request_count": len(rows),
        "common": {
            "k": k, "position_count": position_count,
            "max_new_tokens": 1024, "max_model_len": 4096,
            "temperature": 0.7, "top_p": 1.0,
            "seed": 1, "proxy_top_k": 90,
            "exit_layer": 56, "tree": False,
            "geo_q": GEO_Q,
        },
        "cells": cells,
    }


def main() -> None:
    payload = DATASET.read_bytes()
    dataset_sha = hashlib.sha256(payload).hexdigest()
    if dataset_sha != EXPECTED_SHA:
        raise RuntimeError(f"dataset hash mismatch: {dataset_sha}")
    rows = [json.loads(line) for line in payload.decode().splitlines() if line]
    if len(rows) != 35 or len({row["uid"] for row in rows}) != 35:
        raise RuntimeError("expected the fixed 35-request dataset")
    manifests = ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for k in (9, 8):
        manifest = build(k, rows, dataset_sha)
        path = manifests / f"k{k}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{path}: {len(manifest['cells'])} cells, "
              f"positions={manifest['common']['position_count']}")


if __name__ == "__main__":
    main()
