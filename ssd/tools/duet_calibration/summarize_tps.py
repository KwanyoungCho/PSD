#!/usr/bin/env python3
"""Summarize profiler-OFF DUET K-candidate runs produced by verify_tps.sh."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path


METRICS = {
    "tps": r"Final Decode Throughput:\s*([\d.]+)",
    "tokens_per_step": r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)",
    "p1_hit": r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)",
    "p2_hit": r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)",
    "p1_al": r"Avg Phase 1 Accepted Len:\s*([\d.]+)",
    "p2_al": r"Avg Phase 2 Accepted Len:\s*([\d.]+)",
}
NAME = re.compile(r"^(chain|tree)_k1_(\d+)_k2_(\d+)_seed_(\d+)$")


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def load_runs(root: Path) -> list[dict]:
    rows = []
    for log in sorted(root.glob("*/run.log")):
        match = NAME.match(log.parent.name)
        if not match:
            continue
        text = log.read_text(errors="replace")
        if "EXIT:0" not in text:
            continue
        row = {
            "mode": match.group(1),
            "k1": int(match.group(2)),
            "k2": int(match.group(3)),
            "seed": int(match.group(4)),
            "path": str(log.parent),
        }
        for key, pattern in METRICS.items():
            found = re.search(pattern, text)
            row[key] = float(found.group(1)) if found else None
        if row["tps"] is not None:
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["k1"], row["k2"]), []).append(row)
    out = []
    for (mode, k1, k2), items in sorted(groups.items()):
        summary = {"mode": mode, "k1": k1, "k2": k2, "n": len(items)}
        for key in METRICS:
            values = [x[key] for x in items if x[key] is not None]
            summary[f"{key}_mean"] = st.mean(values) if values else None
            summary[f"{key}_p10"] = _percentile(values, 0.1) if values else None
            summary[f"{key}_p90"] = _percentile(values, 0.9) if values else None
        summary["seeds"] = sorted(x["seed"] for x in items)
        out.append(summary)
    return out


def paired_tps(rows: list[dict]) -> list[dict]:
    candidates = sorted({(x["mode"], x["k1"], x["k2"]) for x in rows})
    result = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if left[0] != right[0]:
                continue
            a = {x["seed"]: x["tps"] for x in rows
                 if (x["mode"], x["k1"], x["k2"]) == left}
            b = {x["seed"]: x["tps"] for x in rows
                 if (x["mode"], x["k1"], x["k2"]) == right}
            common = sorted(a.keys() & b.keys())
            deltas = [b[seed] - a[seed] for seed in common]
            if not deltas:
                continue
            result.append({
                "mode": left[0],
                "left": {"k1": left[1], "k2": left[2]},
                "right": {"k1": right[1], "k2": right[2]},
                "n": len(deltas),
                "right_minus_left_tps_mean": st.mean(deltas),
                "deltas": deltas,
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = load_runs(args.root)
    summary = summarize(rows)
    pairs = paired_tps(rows)
    print("mode  K1 K2  n | TPS mean [p10,p90] | tok/step P1AL P2AL")
    for row in summary:
        print(f"{row['mode']:5s} {row['k1']:2d} {row['k2']:2d} {row['n']:2d} | "
              f"{row['tps_mean']:.2f} [{row['tps_p10']:.2f},{row['tps_p90']:.2f}] | "
              f"{row['tokens_per_step_mean']} {row['p1_al_mean']} {row['p2_al_mean']}")
    if pairs:
        print("\npaired TPS (right - left, same seeds)")
        for pair in pairs:
            left, right = pair["left"], pair["right"]
            print(f"{pair['mode']} {left['k1']}/{left['k2']} -> "
                  f"{right['k1']}/{right['k2']}: "
                  f"{pair['right_minus_left_tps_mean']:+.2f} tok/s "
                  f"(n={pair['n']}, deltas={pair['deltas']})")
    payload = {"runs": rows, "summary": summary, "paired": pairs}
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.strict and not rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
