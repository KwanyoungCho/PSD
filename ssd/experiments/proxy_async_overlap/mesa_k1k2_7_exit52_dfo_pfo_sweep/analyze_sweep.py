#!/usr/bin/env python3
"""MESA K1=K2=7 exit=52 sweep analyzer — 4×3 grid (dfo × pfo)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent
DFO_VALUES = [2, 3, 4, 5]
PFO_VALUES = [1, 2, 3]


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
    # MESA-specific per-phase metrics
    out["p1_hit"] = grab(r"Phase 1.*hit rate:\s*([\d.]+)")
    out["p2_hit"] = grab(r"Phase 2.*hit rate:\s*([\d.]+)")
    out["p1_acceptance"] = grab(r"Phase 1.*acceptance ratio:\s*([\d.]+)")
    out["p2_acceptance"] = grab(r"Phase 2.*acceptance ratio:\s*([\d.]+)")
    out["p1_avg_accepted_len"] = grab(r"Phase 1.*avg accepted length.*:\s*([\d.]+)")
    out["p2_avg_accepted_len"] = grab(r"Phase 2.*avg accepted length.*:\s*([\d.]+)")

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
    head = "| dfo \\ pfo | " + " | ".join(f"pfo={p}" for p in PFO_VALUES) + " |"
    sep = "|---|" + "|".join("---:" for _ in PFO_VALUES) + "|"
    lines.append(head)
    lines.append(sep)
    for D in DFO_VALUES:
        row = [f"dfo={D}"]
        for P in PFO_VALUES:
            v = rows[(D, P)].get(field)
            row.append(cell(v, fmt))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    rows: dict[tuple[int, int], dict] = {}
    for D in DFO_VALUES:
        for P in PFO_VALUES:
            rows[(D, P)] = parse_run(SWEEP_DIR / f"dfo{D}_pfo{P}")

    csv_path = SWEEP_DIR / "sweep_grid.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dfo", "pfo", "f", "decode_tps", "prefill_tps",
                    "target_full_step_ms", "target_verify_ms", "draft_step_ms",
                    "avg_tokens_per_step", "accept_fraction", "cache_hit_rate",
                    "tok_per_step_on_hit", "tok_per_step_on_miss",
                    "p1_hit", "p2_hit", "p1_acceptance", "p2_acceptance",
                    "p1_avg_accepted_len", "p2_avg_accepted_len", "error"])
        for D in DFO_VALUES:
            for P in PFO_VALUES:
                r = rows[(D, P)]
                w.writerow([D, P, D + P,
                            r.get("decode_tps"), r.get("prefill_tps"),
                            r.get("target_full_step_ms"), r.get("target_verify_ms"),
                            r.get("draft_step_ms"), r.get("avg_tokens_per_step"),
                            r.get("accept_fraction"), r.get("cache_hit_rate"),
                            r.get("tok_per_step_on_hit"), r.get("tok_per_step_on_miss"),
                            r.get("p1_hit"), r.get("p2_hit"),
                            r.get("p1_acceptance"), r.get("p2_acceptance"),
                            r.get("p1_avg_accepted_len"), r.get("p2_avg_accepted_len"),
                            r.get("error", "")])
    print(f"-> {csv_path}")

    # Best cell
    best_dp = None
    best_tps = -1.0
    for (D, P), r in rows.items():
        v = r.get("decode_tps")
        if v is not None and v > best_tps:
            best_tps = v
            best_dp = (D, P)

    md = ["# MESA K1=K2=7 exit=52 sweep — dfo × pfo grid (PROFILE_MESA=0)\n"]
    md.append("70B AWQ TP=4 + TinyLlama-1.1B AWQ TP=1, ns=50 in=512 out=512, "
              "seed=42, temp=0.7, --k 14 --mesa_phase1_k 7 --mesa_phase2_k 7, "
              "--mesa_exit_layer 52, SSD_FORCE_SPLIT_K1K2=1, SSD_PROFILE_MESA=0.\n")
    md.append("## decode_tps (tok/s)\n")
    md.append(render_grid(rows, "decode_tps", "{:6.2f}"))
    md.append("\n\n## target_full_step_ms\n")
    md.append(render_grid(rows, "target_full_step_ms", "{:6.2f}"))
    md.append("\n\n## avg_tokens_per_step (incl recovery)\n")
    md.append(render_grid(rows, "avg_tokens_per_step", "{:5.2f}"))
    md.append("\n\n## accept_fraction\n")
    md.append(render_grid(rows, "accept_fraction", "{:5.3f}"))
    md.append("\n\n## cache_hit_rate (Avg Cache Hits)\n")
    md.append(render_grid(rows, "cache_hit_rate", "{:5.3f}"))
    md.append("\n\n## p1_hit (Phase 1 — draft-sourced hit rate)\n")
    md.append(render_grid(rows, "p1_hit", "{:5.3f}"))
    md.append("\n\n## p2_hit (Phase 2 — proxy-sourced hit rate)\n")
    md.append(render_grid(rows, "p2_hit", "{:5.3f}"))
    md.append("\n\n## p1_avg_accepted_len\n")
    md.append(render_grid(rows, "p1_avg_accepted_len", "{:5.2f}"))
    md.append("\n\n## p2_avg_accepted_len\n")
    md.append(render_grid(rows, "p2_avg_accepted_len", "{:5.2f}"))
    md.append("\n\n## draft_step_ms\n")
    md.append(render_grid(rows, "draft_step_ms", "{:6.2f}"))

    if best_dp:
        md.append(f"\n\n## Best decode_tps\n  dfo={best_dp[0]} pfo={best_dp[1]} → **{best_tps:.2f} tok/s**\n")

    fails = [(d, p, r.get("error")) for (d, p), r in rows.items() if r.get("error")]
    if fails:
        md.append("\n## Failed runs\n")
        for D, P, err in fails:
            md.append(f"- dfo={D} pfo={P}: {err}")

    (SWEEP_DIR / "sweep_all.md").write_text("\n".join(md))
    print(f"-> {SWEEP_DIR}/sweep_all.md")

    print()
    print("decode_tps grid:")
    print(render_grid(rows, "decode_tps", "{:6.2f}"))


if __name__ == "__main__":
    main()
