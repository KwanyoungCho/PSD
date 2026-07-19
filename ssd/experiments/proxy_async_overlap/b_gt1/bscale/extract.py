#!/usr/bin/env python3
"""bscale: extract per-cell metrics from run.log files into markdown tables.

Usage: extract.py [cell ...]   (default: the scan-phase cell lists)
Adapted from ../pb_sweep/extract.py (same patterns/format).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

B8_CELLS = [
    "b8_k2x2_d4p1", "b8_k2x2_d5p1", "b8_k3x3_d4p1", "b8_k3x3_d4p2",
    "b8_k4x4_d3p1", "b8_c",
]
B4_CELLS = ["b4_k2x2_d4p1", "b4_k2x2_d5p1", "b4_k3x3_d5p1"]
B1_CELLS = ["b1_e9k24_jit", "b1_c"]

PATTERNS = {
    "decode_tps": r"Final Decode Throughput: ([\d.]+)tok/s",
    "decode_tok_time": r"\[metrics\] Decode tokens/time: (\d+) / ([\d.]+)s",
    "tok_step": r"Avg Tokens per step \(incl recovery\): ([\d.]+)",
    "t_target": r"Avg target time per full step \(ms\): ([\d.]+)",
    "t_verify": r"Avg target verify time \(ms\): ([\d.]+)",
    "cache_hit": r"Avg Cache Hits: ([\d.]+)",
    "p1_hit": r"Avg Phase 1 \(draft\) Hit Rate: ([\d.]+)",
    "p2_hit": r"Avg Phase 2 \(proxy\) Hit Rate: ([\d.]+)",
    "l_p1": r"Avg Phase 1 Accepted Len: ([\d.]+)",
    "l_p2": r"Avg Phase 2 Accepted Len: ([\d.]+)",
    "tok_hit": r"Avg Tokens per step on Cache Hit: ([\d.]+)",
    "tok_miss": r"Avg Tokens per step on Cache Miss: ([\d.]+)",
    "t_draft": r"Avg draft step time \(ms\): ([\d.]+)",
}


def parse(log: Path):
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    out = {"tracebacks": text.count("Traceback")}
    for key, pat in PATTERNS.items():
        m = re.findall(pat, text)
        if not m:
            out[key] = None
        elif key == "decode_tok_time":
            out["decode_tokens"] = int(m[-1][0])
            out["decode_time_s"] = float(m[-1][1])
        else:
            out[key] = float(m[-1])
    return out


def fmt(v, nd=2):
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


METRICS = [
    ("Decode TPS (aggregate)", "decode_tps", 2),
    ("Tokens/step (incl recovery)", "tok_step", 2),
    ("Cache hit rate", "cache_hit", 3),
    ("P1 (draft) hit rate", "p1_hit", 3),
    ("P2 (proxy) hit rate", "p2_hit", 3),
    ("L_p1 (P1 accepted len)", "l_p1", 2),
    ("L_p2 (P2 accepted len)", "l_p2", 2),
    ("Tok/step on hit", "tok_hit", 2),
    ("Tok/step on miss", "tok_miss", 2),
    ("T_target full step (ms)", "t_target", 2),
    ("T_verify (ms)", "t_verify", 2),
    ("T_draft step (ms)", "t_draft", 2),
    ("Decode time (s)", "decode_time_s", 1),
    ("Tracebacks", "tracebacks", 0),
]


def table(cells):
    rows = {c: parse(BASE / c / "run.log") for c in cells}
    hdr = ["metric"] + cells
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for label, key, nd in METRICS:
        vals = [fmt(rows[c].get(key) if rows[c] else None, nd) for c in cells]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    print("\n".join(lines))
    # derived step time t = B*tok_step/TPS
    print("\nDerived (t_step = B*tok/TPS):")
    for c in cells:
        r = rows[c]
        if not r or r.get("decode_tps") is None:
            print(f"  {c}: MISSING")
            continue
        b = int(re.search(r"b(\d+)", c).group(1))
        ts = r.get("tok_step")
        t_step = 1000.0 * b * ts / r["decode_tps"] if ts else None
        print(f"  {c}: t_step {fmt(t_step,1)} ms  per-seq TPS {r['decode_tps']/b:.2f}")


def main():
    cells = sys.argv[1:]
    if cells:
        table(cells)
        return
    print("## B=8 scan\n")
    table(B8_CELLS)
    print("\n## B=4 edge cells\n")
    table(B4_CELLS)
    print("\n## B=1 anchors\n")
    table(B1_CELLS)


if __name__ == "__main__":
    main()
