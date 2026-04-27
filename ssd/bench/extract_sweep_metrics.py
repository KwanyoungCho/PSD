#!/usr/bin/env python3
"""Extract sweep metrics into a markdown table for the final report."""
import os
import re
import sys
from pathlib import Path


def extract(log: Path):
    if not log.exists():
        return None
    text = log.read_text()
    m = {
        "tp": re.search(r"Total Throughput:\s*([\d.]+)", text),
        "accept": re.search(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)", text),
        "tokens_per_step": re.search(r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)", text),
        "cache_hits": re.search(r"Avg Cache Hits:\s*([\d.]+)", text),
        "phase1_hit": re.search(r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)", text),
        "phase2_hit": re.search(r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)", text),
        "draft_ms": re.search(r"Avg draft step time \(ms\):\s*([\d.]+)", text),
        "verify_ms": re.search(r"Avg target verify time \(ms\):\s*([\d.]+)", text),
    }
    return {k: (float(v.group(1)) if v else None) for k, v in m.items()}


def main():
    if len(sys.argv) < 2:
        print("usage: extract_sweep_metrics.py <sweep_dir>")
        sys.exit(1)
    sweep_dir = Path(sys.argv[1])
    rows = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        metrics = extract(d / "run.log")
        if metrics is None or metrics["tp"] is None:
            rows.append((d.name, None))
            continue
        rows.append((d.name, metrics))

    print()
    print("| config | TPS | accept | tok/step | cache_hits | P1_hit | P2_hit | draft_ms | verify_ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in rows:
        if m is None:
            print(f"| {name} | — | — | — | — | — | — | — | — |")
            continue
        def fmt(v, prec=2):
            return f"{v:.{prec}f}" if v is not None else "—"
        print(f"| {name} | **{fmt(m['tp'])}** | {fmt(m['accept'])} | "
              f"{fmt(m['tokens_per_step'])} | {fmt(m['cache_hits'])} | "
              f"{fmt(m['phase1_hit'])} | {fmt(m['phase2_hit'])} | "
              f"{fmt(m['draft_ms'])} | {fmt(m['verify_ms'])} |")

    # Best by TPS
    valid = [(n, m) for n, m in rows if m and m["tp"]]
    if valid:
        valid.sort(key=lambda x: -x[1]["tp"])
        best = valid[0]
        print()
        print(f"**Best by TPS**: {best[0]} → {best[1]['tp']:.2f} tok/s "
              f"(accept={best[1]['accept']:.2f})")


if __name__ == "__main__":
    main()
