#!/usr/bin/env python3
"""sweep/verdict 로그에서 종합 지표 추출 → TSV.

사용: python extract_metrics.py <logdir> [<logdir2> ...]
"""
import re
import sys
import glob
import os

PATTERNS = {
    "decode_tps": r"Final Decode Throughput: ([\d.]+)tok/s",
    "total_tps": r"Total Throughput: ([\d.]+)tok/s",
    "al": r"Avg Tokens per step \(incl recovery\): ([\d.]+)",
    "hit": r"Avg Cache Hits: ([\d.]+)",
    "p1_hit": r"Avg Phase 1 \(draft\) Hit Rate: ([\d.]+)",
    "p2_hit": r"Avg Phase 2 \(proxy\) Hit Rate: ([\d.]+)",
    "p1_al": r"Avg Phase 1 Accepted Len: ([\d.]+)",
    "p2_al": r"Avg Phase 2 Accepted Len: ([\d.]+)",
    "hit_al": r"Avg Tokens per step on Cache Hit: ([\d.]+)",
    "miss_al": r"Avg Tokens per step on Cache Miss: ([\d.]+)",
    "exit": r"EXIT:(\d+)",
}

cols = ["label"] + list(PATTERNS)
print("\t".join(cols))
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        text = open(f, errors="replace").read()
        row = [os.path.basename(f)[:-4]]
        for k, pat in PATTERNS.items():
            m = re.findall(pat, text)
            row.append(m[-1] if m else "-")
        print("\t".join(row))
