#!/usr/bin/env python3
"""Summarize the four-arm formal P1/P2 dynamic-tree gate."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


PATTERNS = {
    "tps": r"Final Decode Throughput:\s*([\d.]+)",
    "tokens_per_step": r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)",
    "cache_hit": r"Avg Cache Hits:\s*([\d.]+)",
    "p1_hit": r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)",
    "p2_hit": r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)",
    "p1_al": r"Avg Phase 1 Accepted Len:\s*([\d.]+)",
    "p2_al": r"Avg Phase 2 Accepted Len:\s*([\d.]+)",
    "hit_tokens": r"Avg Tokens per step on Cache Hit:\s*([\d.]+)",
    "miss_tokens": r"Avg Tokens per step on Cache Miss:\s*([\d.]+)",
    "target_ms": r"Avg target time per full step \(ms\):\s*([\d.]+)",
    "verify_ms": r"Avg target verify time \(ms\):\s*([\d.]+)",
}
ARM_ORDER = ("chain", "p1_tree", "p2_tree", "both")


def read_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def load(root: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(root.glob("*/run_meta.env")):
        run_dir = meta_path.parent
        log_path = run_dir / "run.log"
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="replace")
        meta = read_meta(meta_path)
        row: dict = {
            "server": meta.get("server", "unknown"),
            "arm": meta.get("arm", "unknown"),
            "seed": int(meta.get("seed", -1)),
            "path": str(run_dir),
            "exit": int(re.findall(r"^EXIT:(\d+)$", text, re.MULTILINE)[-1])
                    if re.findall(r"^EXIT:(\d+)$", text, re.MULTILINE) else None,
        }
        for name, pattern in PATTERNS.items():
            match = re.findall(pattern, text)
            row[name] = float(match[-1]) if match else None
        if row["p1_hit"] is not None and row["p1_al"] is not None:
            row["p1_contribution"] = row["p1_hit"] * (row["p1_al"] + 1.0)
        else:
            row["p1_contribution"] = None
        if row["p2_hit"] is not None and row["p2_al"] is not None:
            row["p2_contribution"] = row["p2_hit"] * (row["p2_al"] + 1.0)
        else:
            row["p2_contribution"] = None
        rows.append(row)
    return rows


def mean(values):
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = load(args.root)
    valid = [row for row in rows if row["exit"] == 0 and row["tps"] is not None]
    metrics = ("tps", "tokens_per_step", "cache_hit", "p1_hit", "p1_al",
               "p1_contribution", "p2_hit", "p2_al", "p2_contribution",
               "target_ms", "verify_ms")
    summaries = []
    for arm in ARM_ORDER:
        arm_rows = [row for row in valid if row["arm"] == arm]
        summary = {"arm": arm, "n": len(arm_rows)}
        for metric in metrics:
            summary[metric] = mean(row[metric] for row in arm_rows)
        summary["seeds"] = sorted(row["seed"] for row in arm_rows)
        summaries.append(summary)

    chain_by_pair = {(row["server"], row["seed"]): row for row in valid
                     if row["arm"] == "chain"}
    paired = []
    for arm in ARM_ORDER[1:]:
        deltas = []
        for row in valid:
            if row["arm"] != arm:
                continue
            base = chain_by_pair.get((row["server"], row["seed"]))
            if base is None:
                continue
            entry = {"server": row["server"], "seed": row["seed"]}
            for metric in metrics:
                if row[metric] is not None and base[metric] is not None:
                    entry[metric] = row[metric] - base[metric]
            deltas.append(entry)
        item = {"arm": arm, "n": len(deltas), "runs": deltas}
        for metric in metrics:
            item[metric] = mean(delta.get(metric) for delta in deltas)
        paired.append(item)

    lines = [
        "| arm | n | TPS | tok/step | hit | P1 hit | P1 AL | P1 contribution | P2 hit | P2 AL | P2 contribution | target ms | verify ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def f(key: str) -> str:
            value = row[key]
            return "-" if math.isnan(value) else f"{value:.3f}"
        lines.append(
            f"| {row['arm']} | {row['n']} | {f('tps')} | {f('tokens_per_step')} | "
            f"{f('cache_hit')} | {f('p1_hit')} | {f('p1_al')} | "
            f"{f('p1_contribution')} | {f('p2_hit')} | {f('p2_al')} | "
            f"{f('p2_contribution')} | {f('target_ms')} | {f('verify_ms')} |")
    lines += ["", "Paired mean delta versus chain on the same server and seed:", ""]
    lines += [
        "| arm - chain | n | TPS | tok/step | hit | P1 hit | P1 AL | P2 hit | P2 AL | target ms | verify ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        def d(key: str) -> str:
            value = row[key]
            return "-" if math.isnan(value) else f"{value:+.3f}"
        lines.append(
            f"| {row['arm']} | {row['n']} | {d('tps')} | {d('tokens_per_step')} | "
            f"{d('cache_hit')} | {d('p1_hit')} | {d('p1_al')} | "
            f"{d('p2_hit')} | {d('p2_al')} | {d('target_ms')} | {d('verify_ms')} |")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.markdown_out:
        args.markdown_out.write_text(report)
    if args.json_out:
        args.json_out.write_text(json.dumps({"runs": rows, "summary": summaries,
                                             "paired_vs_chain": paired},
                                            indent=2) + "\n")
    if args.strict:
        expected = {(arm, seed) for arm in ARM_ORDER for seed in (42, 123, 2024)}
        observed = {(row["arm"], row["seed"]) for row in valid}
        if not expected.issubset(observed):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
