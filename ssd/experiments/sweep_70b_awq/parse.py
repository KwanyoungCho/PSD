"""Parse a single bench run.log → metrics dict.

Usage: python parse.py <run.log>  (prints JSON one-liner to stdout)
"""
import json
import re
import sys
from pathlib import Path


PATTERNS = {
    "throughput": r"Total Throughput:\s*([\d.]+)",
    "tok_per_step": r"Avg Tokens per step\s*[:=]\s*([\d.]+)",
    "tok_per_step_hit": r"Avg Tokens per step on Cache Hit:\s*([\d.]+)",
    "tok_per_step_miss": r"Avg Tokens per step on Cache Miss:\s*([\d.]+)",
    "accept": r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)",
    "cache_hit": r"Avg Cache Hits:\s*([\d.]+)",
    "draft_ms": r"Avg draft step\s*\(ms\):\s*([\d.]+)",
    "verify_ms": r"Avg target verify\s*\(ms\):\s*([\d.]+)",
}


def parse(log_path: Path) -> dict:
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    out = {"path": str(log_path)}
    for k, pat in PATTERNS.items():
        m = re.search(pat, text)
        out[k] = float(m.group(1)) if m else None
    out["completed"] = "Engine exited" in text or out["throughput"] is not None
    return out


if __name__ == "__main__":
    p = Path(sys.argv[1])
    print(json.dumps(parse(p)))
