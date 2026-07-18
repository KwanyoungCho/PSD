#!/usr/bin/env python3
"""M5 sweep: extract per-cell metrics from run.log files into a markdown table."""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CELLS = [f"{sys_}_b{b}" for b in (1, 2, 4) for sys_ in ("duet", "c")]

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


def main():
    rows = {c: parse(BASE / c / "run.log") for c in CELLS}
    hdr = ["metric"] + CELLS
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    metrics = [
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
    for label, key, nd in metrics:
        vals = [fmt(rows[c].get(key) if rows[c] else None, nd) for c in CELLS]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    print("\n".join(lines))

    # derived: per-seq TPS, miss share, any-miss burden
    print("\nDerived:")
    for c in CELLS:
        r = rows[c]
        if not r or r.get("decode_tps") is None:
            print(f"  {c}: MISSING")
            continue
        b = int(c.rsplit("b", 1)[1])
        hit = r.get("cache_hit")
        per_seq = r["decode_tps"] / b
        miss = 1 - hit if hit is not None else None
        print(
            f"  {c}: per-seq TPS {per_seq:.2f}  miss share {fmt(miss,3)}  "
            f"step time proxy: target {fmt(r.get('t_target'))} ms, "
            f"draft {fmt(r.get('t_draft'))} ms"
        )


if __name__ == "__main__":
    main()
