#!/usr/bin/env python3
"""Strictly validate and aggregate a K-specific cache-budget sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ("duet", "only_proxy", "geo", "uniform")
BUDGETS = tuple(range(2, 9))


def weighted_hit(rows: list[dict]) -> tuple[float, int]:
    total = 0.0
    steps = 0
    for row in rows:
        rate = row.get("cache_hit_rate")
        count = row.get("n_verify_steps")
        if rate is not None and count is not None:
            total += float(rate) * int(count)
            steps += int(count)
    if not steps:
        raise RuntimeError("no verification steps")
    return total / steps, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", required=True, type=int, choices=(8, 9))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    k = args.k
    manifest = json.loads((ROOT / "manifests" / f"k{k}.json").read_text())
    results = ROOT / "results" / f"k{k}_seed1"
    expected_uids = {
        json.loads(line)["uid"]
        for line in Path(manifest["dataset"]).read_text().splitlines() if line
    }
    indexed = {
        (cell["method"], cell["avg_position_budget"]): cell
        for cell in manifest["cells"]
    }
    summaries = []
    errors = []
    for method in METHODS:
        for budget in BUDGETS:
            cell = indexed[(method, budget)]
            path = results / "raw" / f"{method}_b{budget}.jsonl"
            rows = ([json.loads(line) for line in path.read_text().splitlines()
                     if line] if path.exists() else [])
            got = [row.get("uid") for row in rows]
            complete = (len(rows) == 35 and set(got) == expected_uids
                        and len(set(got)) == 35)
            if complete:
                for row in rows:
                    if row.get("seed") != 1 or row.get("sampler_seed") != 1:
                        errors.append(f"{method}/b{budget}: seed mismatch")
                    if int(row.get("position_count", -1)) != k + 1:
                        errors.append(f"{method}/b{budget}: position mismatch")
                    if int(row.get("cache_root_budget", -1)) != cell["total_cache_roots"]:
                        errors.append(f"{method}/b{budget}: total-root mismatch")
                    if method in ("duet", "only_proxy") and row.get("proxy_top_k") != 90:
                        errors.append(f"{method}/b{budget}: proxy-top-k mismatch")
                    if method in ("geo", "uniform"):
                        if row.get("engine") != "ssd-dedicated(eager)":
                            errors.append(
                                f"{method}/b{budget}: unexpected SSD mode")
                        if row.get("engine_root") != "/home/eslab/chokwans99/ssd":
                            errors.append(
                                f"{method}/b{budget}: not SSD dedicated root")
                        if row.get("jit_speculate") is not True:
                            errors.append(
                                f"{method}/b{budget}: JIT speculation disabled")
                hit, steps = weighted_hit(rows)
            else:
                errors.append(f"{method}/b{budget}: {len(set(got))}/35 requests")
                hit, steps = (None, 0)
            summaries.append({
                "k": k, "position_count": k + 1, "method": method,
                "avg_position_budget": budget,
                "total_cache_roots": cell["total_cache_roots"],
                "weighted_cache_hit_rate": hit,
                "verify_steps": steps, "requests": len(rows),
                "complete": complete,
            })
    results.mkdir(parents=True, exist_ok=True)
    with (results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if errors and not args.allow_incomplete:
        raise SystemExit("; ".join(dict.fromkeys(errors)))
    report = [
        f"# Cache-budget sweep: K={k}, positions={k + 1}", "",
        "Actual sampler seed: 1", "DUET/Only-Proxy proxy top-k: 90",
        "Dataset: fixed 35-request Spec-Bench subset",
        "Aggregation: verification-step-weighted hit rate", "",
    ]
    report.append("All 28 cells complete." if not errors else "Incomplete/errors:")
    report.extend(f"- {error}" for error in dict.fromkeys(errors))
    (results / "REPORT.md").write_text("\n".join(report) + "\n")
    print(results / "summary.csv")


if __name__ == "__main__":
    main()
