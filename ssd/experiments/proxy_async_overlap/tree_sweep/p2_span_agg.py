#!/usr/bin/env python3
"""P2 구간 집계기 (리뷰6 교정판 — docs/duet/22).

교정: hit_k2 스텝은 TREE_GLUE가 phase2_prep/replay 라벨을 공유해
rollout 4개가 아니라 5개로 세졌다 (pre 음수 증상). phase2_build.start
이후의 **처음 4개 replay**만 rollout으로 취한다.

사용: python p2_span_agg.py <draft_profile.json> [skip_steps=20]
출력: step 평균 / pre / 4-fwd 창 / prep합 / replay합 / 간격 / post
"""
import json
import sys
from collections import defaultdict


def p50(v):
    s = sorted(v)
    return s[len(s) // 2] if s else float("nan")


def agg(path, skip=20):
    rows = [r for r in json.load(open(path)) if r.get("label") != "_anchor"]
    st = {r["step_id"]: r["status"] for r in rows
          if r.get("status") and r.get("step_id") is not None}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["label"] in ("phase2_build", "phase2_prep", "phase2_replay",
                          "merge_cache"):
            by[r["step_id"]][r["label"]].append(
                (r["start_ms"], r["end_ms"], r.get("ms", 0)))
    out = defaultdict(list)
    for sid, d in by.items():
        if "phase2_build" not in d:
            continue
        b0 = min(x[0] for x in d["phase2_build"])
        reps = sorted(x for x in d.get("phase2_replay", []) if x[0] > b0)
        preps = sorted(x for x in d.get("phase2_prep", []) if x[0] > b0)
        if len(reps) < 4:
            continue
        reps = reps[:4]                      # rollout 4개만 (glue 배제)
        last_end = reps[3][1]
        preps = [x for x in preps if x[0] < last_end]
        allf = sorted(reps + preps)
        out["wall"].append(last_end - allf[0][0])
        out["pre"].append(allf[0][0] - b0)
        out["gap"].append(sum(max(0.0, b[0] - a[1])
                              for a, b in zip(allf, allf[1:])))
        out["prep_sum"].append(sum(x[2] for x in preps))
        out["replay_sum"].append(sum(x[2] for x in reps))
        mrg = by.get(sid + 1, {}).get("merge_cache")
        if mrg:
            out["post"].append(max(x[1] for x in mrg) - last_end)
    # step 주기
    smin = {}
    for r in rows:
        sid, sm = r.get("step_id"), r.get("start_ms")
        if sid is None or sm is None:
            continue
        smin[sid] = min(smin.get(sid, 1e18), sm)
    ks = sorted(smin)
    per = [smin[b] - smin[a] for a, b in zip(ks, ks[1:])
           if b == a + 1 and a > skip]
    return out, per


if __name__ == "__main__":
    out, per = agg(sys.argv[1],
                   int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    print(f"step 평균 {sum(per)/len(per):.2f} p50 {p50(per):.2f} "
          f"(n={len(per)})")
    for k in ("pre", "wall", "gap", "prep_sum", "replay_sum", "post"):
        v = out[k]
        print(f"{k:11s} p50 {p50(v):7.3f}  mean "
              f"{sum(v)/len(v) if v else float('nan'):7.3f}  n={len(v)}")
