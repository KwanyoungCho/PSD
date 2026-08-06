"""P2 구간 비교: arena vs exec 프로파일 (draft JSON).

step별 phase2_* 구간의 [최초 start, 최후 end] span, GPU busy 합,
idle(span-busy), CPU dispatch 시간 합 → p50/p95 비교.
"""
import json, sys, glob, statistics


def load(dirpat):
    fs = sorted(glob.glob(dirpat + "/duet_profile_draft_*.json"))
    assert fs, dirpat
    return json.load(open(fs[-1]))


def p2_spans(recs):
    by_step = {}
    for r in recs:
        lab = r.get("label") or ""
        if not lab.startswith("phase2"):
            continue
        sid = r.get("step_id")
        if sid is None:
            continue
        by_step.setdefault(sid, []).append(r)
    spans, busys, idles, cpus = [], [], [], []
    for sid, rs in by_step.items():
        s = min(x["start_ms"] for x in rs)
        e = max(x["end_ms"] for x in rs)
        busy = sum(x["ms"] for x in rs)
        cpu = sum((x.get("cpu_dispatch_end_ns", 0)
                   - x.get("cpu_dispatch_start_ns", 0)) for x in rs) / 1e6
        spans.append(e - s)
        busys.append(busy)
        idles.append(max(0.0, (e - s) - busy))
        cpus.append(cpu)
    return spans, busys, idles, cpus


def q(v, p):
    return statistics.quantiles(v, n=100)[p - 1]


for name, d in (("arena", sys.argv[1]), ("exec", sys.argv[2])):
    sp, bu, idl, cp = p2_spans(load(d))
    print(f"{name}: n={len(sp)}  span p50={q(sp,50):.3f} "
          f"p95={q(sp,95):.3f} ms | busy p50={q(bu,50):.3f} | "
          f"idle p50={q(idl,50):.3f} p95={q(idl,95):.3f} | "
          f"CPU-dispatch p50={q(cp,50):.3f} ms")
