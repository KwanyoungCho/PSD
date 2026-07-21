#!/usr/bin/env python3
"""bscale32: extract per-cell metrics from run.log files into markdown tables.

Usage: extract.py [cell ...]   (default: the Phase A + Phase B cell lists)
Adapted from ../bscale/extract.py (same patterns/format).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

C_CELLS = {
    2: ["cb2_k7f6", "cb2_k5f6", "cb2_k3f6", "cb2_k5f3", "cb2_k3f3"],
    4: ["cb4_k7f6", "cb4_k5f6", "cb4_k3f6", "cb4_k5f3", "cb4_k3f3"],
    8: ["cb8_k7f6", "cb8_k5f6", "cb8_k3f6", "cb8_k5f3", "cb8_k3f3"],
    16: ["cb16_k7f6", "cb16_k5f6", "cb16_k3f6", "cb16_k5f3", "cb16_k3f3"],
    32: ["cb32_k3f6", "cb32_k5f6", "cb32_k5f3", "cb32_k3f3", "cb32_k2f3",
         "cb32_k2f2"],
}
DUET_CELLS = {
    16: ["b16_k2x2_d5p1", "b16_k2x2_d4p1", "b16_k3x3_d4p1", "b16_k1x1_d5p1"],
    32: ["b32_k2x2_d5p1", "b32_k2x2_d4p1", "b32_k1x1_d5p1", "b32_k1x1_d7p1"],
}

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
    out = {
        "tracebacks": text.count("Traceback"),
        "oom": ("OutOfMemoryError" in text) or ("CUDA out of memory" in text),
    }
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
    print("\nDerived (t_step = B*tok/TPS):")
    for c in cells:
        r = rows[c]
        if not r:
            print(f"  {c}: MISSING")
            continue
        if r.get("decode_tps") is None:
            tag = "DNF(OOM)" if r.get("oom") else "NO_TPS"
            print(f"  {c}: {tag} tracebacks={r['tracebacks']}")
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
    for b, cl in C_CELLS.items():
        print(f"\n## Phase A — C scan B={b}\n")
        table(cl)
    for b, cl in DUET_CELLS.items():
        print(f"\n## Phase B — DUET scan B={b}\n")
        table(cl)


if __name__ == "__main__":
    main()
