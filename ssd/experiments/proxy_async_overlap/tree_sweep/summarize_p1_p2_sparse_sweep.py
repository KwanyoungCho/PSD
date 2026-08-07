#!/usr/bin/env python3
"""Summarize sparse DUET sweep outputs without mixing failed runs.

Usage:
  python summarize_p1_p2_sparse_sweep.py /path/to/sweep_root

Prints one row per run followed by a mean row per (label, arm).  Only logs
ending in EXIT:0 and containing all headline metrics are included.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


PATTERNS = {
    "tps": r"Final Decode Throughput:\s*([0-9.]+)",
    "tok_step": r"Avg Tokens per step \(incl recovery\):\s*([0-9.]+)",
    "cache_hit": r"Avg Cache Hits:\s*([0-9.]+)",
    "p1_hit": r"Avg Phase 1 \(draft\) Hit Rate:\s*([0-9.]+)",
    "p2_hit": r"Avg Phase 2 \(proxy\) Hit Rate:\s*([0-9.]+)",
    "p1_al": r"Avg Phase 1 Accepted Len:\s*([0-9.]+)",
    "p2_al": r"Avg Phase 2 Accepted Len:\s*([0-9.]+)",
    "target_ms": r"Avg target time per full step \(ms\):\s*([0-9.]+)",
    "draft_ms": r"Avg draft step time \(ms\):\s*([0-9.]+)",
}


def env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def one_run(log: Path) -> dict[str, object] | None:
    text = log.read_text(errors="replace")
    if not text.rstrip().endswith("EXIT:0"):
        return None
    values: dict[str, object] = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        if match is None:
            return None
        values[key] = float(match.group(1))
    meta_path = log.parent / "run_meta.env"
    meta = env_file(meta_path) if meta_path.exists() else {}
    values.update({
        "label": log.parent.parent.name,
        "arm": meta.get("arm", "?"),
        "seed": meta.get("seed", "?"),
        "k1": meta.get("k1", "?"),
        "k2": meta.get("k2", "?"),
        "n1": meta.get("p1_tree_max_nodes", "?"),
        "n2": meta.get("p2_tree_max_nodes", "?"),
        "c": meta.get("tree_c", "?"),
        "p1_scale": meta.get("p1_tree_forward_scale", "?"),
        "p2_width": meta.get("p2_width", "?"),
        "p1exec": "p1exec stats:" in text,
        "p2exec": "p2exec stats:" in text,
        "fallback": "fallback" in text or "unsupported_cfg" in text,
    })
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = [row for log in sorted(args.root.rglob("run.log"))
            if (row := one_run(log)) is not None]
    numeric = list(PATTERNS)
    header = ["kind", "label", "arm", "seed", "k1", "k2", "n1", "n2",
              "c", "p1_scale", "p2_width", *numeric,
              "p1exec", "p2exec", "fallback"]
    print("\t".join(header))

    def emit(kind: str, row: dict[str, object]) -> None:
        vals = [kind] + [row.get(k, "") for k in header[1:]]
        print("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                        for v in vals))

    for row in rows:
        emit("run", row)

    groups: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["label"], row["arm"]), []).append(row)
    for (_, _), group in sorted(groups.items()):
        mean = dict(group[0])
        mean["seed"] = f"mean(n={len(group)})"
        for key in numeric:
            mean[key] = statistics.fmean(float(x[key]) for x in group)
        mean["p1exec"] = all(bool(x["p1exec"]) for x in group)
        mean["p2exec"] = all(bool(x["p2exec"]) for x in group)
        mean["fallback"] = any(bool(x["fallback"]) for x in group)
        emit("mean", mean)


if __name__ == "__main__":
    main()
