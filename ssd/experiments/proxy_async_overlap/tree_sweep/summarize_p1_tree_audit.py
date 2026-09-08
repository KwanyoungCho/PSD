#!/usr/bin/env python3
"""Step-weighted summary for run_duet.py P1 tree audit outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def weighted(rows: list[dict], key: str, weight) -> float | None:
    pairs = []
    for row in rows:
        value = row.get(key)
        w = weight(row)
        if value is not None and w is not None and w > 0:
            pairs.append((float(value), float(w)))
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def summarize(path: Path) -> dict:
    rows = load(path)
    steps = lambda r: r.get("n_verify_steps") or 0
    p1_steps = lambda r: (r.get("n_verify_steps") or 0) * \
        (r.get("p1_hit_rate") or 0)
    p2_steps = lambda r: (r.get("n_verify_steps") or 0) * \
        (r.get("p2_hit_rate") or 0)
    total_time = sum(float(r.get("decode_total_time") or 0) for r in rows)
    total_tokens = sum(int(r.get("decode_total_tokens") or 0) for r in rows)
    p1_hit = weighted(rows, "p1_hit_rate", steps)
    p1_al = weighted(rows, "p1_accept_len", p1_steps)
    p2_hit = weighted(rows, "p2_hit_rate", steps)
    p2_al = weighted(rows, "p2_accept_len", p2_steps)
    wall_time = sum(float(r.get("wall_s") or 0) for r in rows)
    completion_tokens = sum(int(r.get("completion_tokens") or 0)
                            for r in rows)
    return {
        "arm": path.stem,
        "requests": len(rows),
        "verify_steps": sum(int(steps(r)) for r in rows),
        "decode_tps": total_tokens / total_time if total_time else None,
        "wall_tps": completion_tokens / wall_time if wall_time else None,
        "tokens_per_step": weighted(rows, "accept_len", steps),
        "cache_hit_rate": weighted(rows, "cache_hit_rate", steps),
        "p1_hit_rate": p1_hit,
        "p1_hit_steps": sum(float(p1_steps(r)) for r in rows),
        "p1_accept_len": p1_al,
        "p1_hit_x_accept": (p1_hit * p1_al
                             if p1_hit is not None and p1_al is not None
                             else None),
        "p2_hit_rate": p2_hit,
        "p2_hit_steps": sum(float(p2_steps(r)) for r in rows),
        "p2_accept_len": p2_al,
        "target_step_ms": ((weighted(rows, "mean_target_step_s", steps)
                            or 0) * 1000),
        "target_verify_ms": ((weighted(
            rows, "mean_target_verify_s", steps) or 0) * 1000),
        "completion_tokens": completion_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    out = [summarize(path) for path in args.inputs]
    if not out:
        raise SystemExit("no inputs")
    fields = list(out[0])
    print("\t".join(fields))
    for row in out:
        print("\t".join(
            "" if row[k] is None else
            (f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]))
            for k in fields))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out)


if __name__ == "__main__":
    main()
