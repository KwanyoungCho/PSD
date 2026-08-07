#!/usr/bin/env python3
"""트리 PROFILE 런 스팬 집계 — status(hit_k2=트리 스텝)별 label 평균/합.

사용: python span_agg.py <profile_dir>
"""
import json, sys, glob
from collections import defaultdict


def load(d, role):
    fs = sorted(glob.glob(f"{d}/duet_profile_{role}_*.json"))
    assert fs, f"no {role} json in {d}"
    return json.load(open(fs[-1]))


def agg(rows, tag):
    # 시간 필드 탐색 (start/end ns or ms)
    t = defaultdict(lambda: [0.0, 0])
    step_status = {}
    for r in rows:
        if r.get("step_id") is not None and r.get("status"):
            step_status[r["step_id"]] = r["status"]
    for r in rows:
        lb = r.get("label")
        if not lb or lb.startswith("_"):
            continue
        dur = None
        for a, b in (("start_ns", "end_ns"), ("start", "end"),
                     ("t0_ns", "t1_ns")):
            if a in r and b in r and r[a] is not None and r[b] is not None:
                dur = (r[b] - r[a]) / 1e6
                break
        if dur is None and "dur_ms" in r:
            dur = r["dur_ms"]
        if dur is None and "cuda_ms" in r:
            dur = r["cuda_ms"]
        if dur is None:
            continue
        st = step_status.get(r.get("step_id"), r.get("status") or "?")
        key = (st, lb)
        t[key][0] += dur
        t[key][1] += 1
    print(f"\n==== {tag} ====")
    by_status = defaultdict(list)
    for (st, lb), (s, n) in t.items():
        by_status[st].append((s / max(n, 1), s, n, lb))
    for st in sorted(by_status):
        print(f"\n-- status={st} --")
        for mean, total, n, lb in sorted(by_status[st], reverse=True)[:14]:
            print(f"  {lb:<28} mean {mean:8.2f} ms  x{n:<5} total {total/1000:8.2f} s")


if __name__ == "__main__":
    d = sys.argv[1]
    if any(True for _ in glob.iglob(f"{d}/duet_profile_target_rank0_*.json")):
        agg(load(d, "target_rank0"), "target_rank0")
    agg(load(d, "draft"), "draft")
