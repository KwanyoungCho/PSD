#!/usr/bin/env python3
"""Async SD sweep analyzer — produces a 4×4 TPS grid from per-run logs.

Reads each `k{K}_f{F}/run.log`, extracts headline metrics, writes:
  - sweep_grid.csv         — long format (one row per (k, f) cell)
  - sweep_decode_tps.md     — 4×4 markdown grid of decode_tps
  - sweep_all.md            — 4×4 grids for decode_tps, accept_frac,
                              cache_hit, tok_per_step, target_full_step_ms
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent
K_VALUES = [7, 8, 9, 10]
F_VALUES = [3, 4, 5, 6]


def parse_run(outdir: Path) -> dict:
    log = outdir / "run.log"
    if not log.exists():
        return {"error": "no run.log"}
    text = log.read_text()
    out: dict = {}

    def grab(pat: str, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None

    out["decode_tps"] = grab(r"Final Decode Throughput:\s*([\d.]+)tok/s")
    out["prefill_tps"] = grab(r"Final Prefill Throughput:\s*([\d.]+)tok/s")
    out["target_full_step_ms"] = grab(r"Avg target time per full step \(ms\):\s*([\d.]+)")
    out["target_verify_ms"] = grab(r"Avg target verify time \(ms\):\s*([\d.]+)")
    out["draft_step_ms"] = grab(r"Avg draft step time \(ms\):\s*([\d.]+)")
    out["avg_tokens_per_step"] = grab(r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)")
    out["accept_fraction"] = grab(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)")
    out["cache_hit_rate"] = grab(r"Avg Cache Hits:\s*([\d.]+)")
    out["tok_per_step_on_hit"] = grab(r"Tokens per step on hit:\s*([\d.]+)")
    out["tok_per_step_on_miss"] = grab(r"Tokens per step on miss:\s*([\d.]+)")

    # Failure detection — bench prints Traceback or our script's FAILED marker.
    if "=== FAILED" in text:
        out["error"] = "run failed (see log)"
    elif out["decode_tps"] is None:
        out["error"] = "decode_tps not parsed"

    return out


def cell(v, fmt="{:6.2f}"):
    if v is None:
        return "  —   "
    if isinstance(v, str):
        return v
    return fmt.format(v)


def render_grid(rows: dict[tuple[int, int], dict], field: str, fmt="{:6.2f}") -> str:
    lines = []
    head = "| k \\ f | " + " | ".join(f" f={f} " for f in F_VALUES) + " |"
    sep = "|---|" + "|".join("---:" for _ in F_VALUES) + "|"
    lines.append(head)
    lines.append(sep)
    for K in K_VALUES:
        row = [f"k={K}"]
        for F in F_VALUES:
            v = rows[(K, F)].get(field)
            row.append(cell(v, fmt))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    rows: dict[tuple[int, int], dict] = {}
    for K in K_VALUES:
        for F in F_VALUES:
            rows[(K, F)] = parse_run(SWEEP_DIR / f"k{K}_f{F}")

    # CSV
    csv_path = SWEEP_DIR / "sweep_grid.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "f", "decode_tps", "prefill_tps", "target_full_step_ms",
                    "target_verify_ms", "draft_step_ms", "avg_tokens_per_step",
                    "accept_fraction", "cache_hit_rate",
                    "tok_per_step_on_hit", "tok_per_step_on_miss", "error"])
        for K in K_VALUES:
            for F in F_VALUES:
                r = rows[(K, F)]
                w.writerow([K, F, r.get("decode_tps"), r.get("prefill_tps"),
                            r.get("target_full_step_ms"), r.get("target_verify_ms"),
                            r.get("draft_step_ms"), r.get("avg_tokens_per_step"),
                            r.get("accept_fraction"), r.get("cache_hit_rate"),
                            r.get("tok_per_step_on_hit"), r.get("tok_per_step_on_miss"),
                            r.get("error", "")])
    print(f"-> {csv_path}")

    # Markdown — decode TPS focus
    md1 = ["# Async SD sweep — decode_tps (PROFILE_MESA=0, no measurement overhead)\n"]
    md1.append("70B AWQ TP=4 + TinyLlama AWQ TP=1, ns=50 in=512 out=512, seed=42, temp=0.7.\n")
    md1.append("## decode_tps (tok/s)\n")
    md1.append(render_grid(rows, "decode_tps", "{:6.2f}"))
    (SWEEP_DIR / "sweep_decode_tps.md").write_text("\n".join(md1))
    print(f"-> {SWEEP_DIR}/sweep_decode_tps.md")

    # Markdown — all metrics
    md = ["# Async SD sweep — full grid (PROFILE_MESA=0)\n"]
    md.append("Same config as sweep_decode_tps.md; values are means over decode.\n")
    md.append("\n## decode_tps (tok/s)\n")
    md.append(render_grid(rows, "decode_tps", "{:6.2f}"))
    md.append("\n\n## target_full_step_ms\n")
    md.append(render_grid(rows, "target_full_step_ms", "{:6.2f}"))
    md.append("\n\n## avg_tokens_per_step (incl recovery)\n")
    md.append(render_grid(rows, "avg_tokens_per_step", "{:5.2f}"))
    md.append("\n\n## accept_fraction\n")
    md.append(render_grid(rows, "accept_fraction", "{:5.3f}"))
    md.append("\n\n## cache_hit_rate\n")
    md.append(render_grid(rows, "cache_hit_rate", "{:5.3f}"))
    md.append("\n\n## draft_step_ms\n")
    md.append(render_grid(rows, "draft_step_ms", "{:6.2f}"))

    # Highlight best
    best_kf = None
    best_tps = -1.0
    for (K, F), r in rows.items():
        v = r.get("decode_tps")
        if v is not None and v > best_tps:
            best_tps = v
            best_kf = (K, F)
    if best_kf:
        md.append(f"\n\n## Best decode_tps\n")
        md.append(f"  k={best_kf[0]} f={best_kf[1]} → **{best_tps:.2f} tok/s**\n")

    # Failed runs
    fails = [(k, f, r.get("error")) for (k, f), r in rows.items() if r.get("error")]
    if fails:
        md.append("\n\n## Failed runs\n")
        for K, F, err in fails:
            md.append(f"- k={K} f={F}: {err}")

    (SWEEP_DIR / "sweep_all.md").write_text("\n".join(md))
    print(f"-> {SWEEP_DIR}/sweep_all.md")

    # Print compact summary to stdout
    print()
    print("decode_tps grid:")
    print(render_grid(rows, "decode_tps", "{:6.2f}"))


if __name__ == "__main__":
    main()
