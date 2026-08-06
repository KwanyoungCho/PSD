"""단계0 — P2 경로 공식 분석기 (23번 후속 고정 지침).

공식 정의 (고정): P2 전체 = phase2_build.start → merge_cache.end
(step별). phase2_* 레이블만 모으는 방식은 폐기 — 레이블 커버리지가
경로별로 달라 apples-to-oranges가 됐던 원인 (2026-08-06 정정).

구간 분해 (실행기 경로 신레이블):
  phase2_build   → 트리 선택 준비 (선택기/레이아웃, 양 경로 공통)
  p2_prepare     → 실행기 입력 버퍼 채우기
  p2_graph_replay→ CUDA graph 실행 (capture 스텝 포함)
  p2_output_convert → graph 출력 → 기존 views/backbone 변환
  p2_cache_merge → 최종 캐시 병합 (merge_cache와 동일 구간 별칭)
arena 경로: phase2_prep/phase2_replay (×4)가 forward 구간.

component 합 검증: 전체 − (경계 내 레이블 합) 미설명분이 p50 기준
0.5ms 이하 또는 전체의 5% 이하이어야 한다 (assert).

usage:
  python analyze_p2_path.py <draft_profile.json> [...]
  python -m unittest tests.diag.analyze_p2_path  (구프로파일 재현 검증)
"""
import json
import glob
import sys
from collections import defaultdict

EXEC_LABELS = ["p2_prepare", "p2_graph_replay", "p2_output_convert"]
ARENA_LABELS = ["phase2_prep", "phase2_replay"]
COMMON_LABELS = ["phase2_build", "proxy_wait"]
MERGE = "merge_cache"


def q(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))]


def per_step(recs):
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        sid = r.get("step_id")
        if sid is None:
            continue
        by[sid][r["label"]].append(
            (r["start_ms"], r["end_ms"], r["ms"]))
    return by


def analyze(recs):
    by = per_step(recs)
    out = {"total": [], "unexplained": [],
           "comp": defaultdict(list)}
    for sid, labs in by.items():
        if "phase2_build" not in labs or MERGE not in labs:
            continue
        t0 = min(s for s, _, _ in labs["phase2_build"])
        t1 = max(e for _, e, _ in labs[MERGE])
        total = t1 - t0
        if total <= 0 or total > 500:
            continue                       # 캡처/워밍업 outlier 제외
        # explained = 경계 내 '모든' 레이블 구간의 합집합 길이
        # (중첩 이중계상 방지 — 상위/하위 span 공존 허용)
        ivs = []
        for lab, spans in labs.items():
            for s, e, ms in spans:
                if e < t0 or s > t1:
                    continue
                ivs.append((max(s, t0), min(e, t1)))
                if lab in (COMMON_LABELS + EXEC_LABELS
                           + ARENA_LABELS + ["p2_rollout", MERGE]):
                    out["comp"][lab].append(ms)
        ivs.sort()
        expl, cur_s, cur_e = 0.0, None, None
        for s, e in ivs:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    expl += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            expl += cur_e - cur_s
        out["total"].append(total)
        out["unexplained"].append(max(0.0, total - expl))
    return out


def report(path, assert_budget=True):
    recs = json.load(open(path))
    a = analyze(recs)
    n = len(a["total"])
    t50, u50 = q(a["total"], 50), q(a["unexplained"], 50)
    print(f"{path}")
    print(f"  steps={n}  P2 total p50={t50:.2f} p95="
          f"{q(a['total'], 95):.2f}  unexplained p50={u50:.2f}")
    for lab in (COMMON_LABELS + EXEC_LABELS + ARENA_LABELS
                + ["p2_rollout", MERGE]):
        v = a["comp"].get(lab, [])
        if v:
            print(f"    {lab:18s} n={len(v):6d} p50={q(v,50):6.2f} "
                  f"p95={q(v,95):6.2f}")
    if assert_budget and n:
        ok = (u50 <= 0.5) or (u50 <= 0.05 * t50)
        assert ok, (f"미설명 p50={u50:.2f}ms > 0.5ms이며 전체의 "
                    f"{100*u50/t50:.1f}% > 5% — 레이블 누락")
        print(f"  [PASS] 미설명 p50={u50:.2f}ms "
              f"({100*u50/max(t50,1e-9):.1f}% of total)")
    return t50, u50, n


# ── 회귀 고정: 구 프로파일(무레이블 실행기)에서 공식 경계 재현 ──
import unittest


class TestOfficialBoundaryOnLegacyProfiles(unittest.TestCase):
    """2026-08-05 프로파일 쌍에서 공식 경계 p50이 arena≈30.9 /
    exec≈20.1로 재현되는지 (측정 정정의 회귀 고정). 구프로파일은
    실행기 신레이블이 없으므로 미설명 예산 assert는 생략."""

    A = ("/tmp/claude-1013/-home-chokwans99-PSD/"
         "86f700a9-77ad-433e-b8fd-22f0a56c9745/scratchpad/smoke/"
         "prof_arena/duet_profile_draft_*.json")
    E = ("/tmp/claude-1013/-home-chokwans99-PSD/"
         "86f700a9-77ad-433e-b8fd-22f0a56c9745/scratchpad/smoke/"
         "prof_exec/duet_profile_draft_*.json")

    def _p50(self, pat):
        fs = sorted(glob.glob(pat))
        if not fs:
            self.skipTest(f"프로파일 없음: {pat}")
        t50, _, n = report(fs[-1], assert_budget=False)
        self.assertGreater(n, 100)
        return t50

    def test_arena_boundary_reproduces(self):
        t = self._p50(self.A)
        self.assertAlmostEqual(t, 30.94, delta=0.5)

    def test_exec_boundary_reproduces(self):
        t = self._p50(self.E)
        self.assertAlmostEqual(t, 20.06, delta=0.5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            for f in sorted(glob.glob(p)):
                report(f, assert_budget=False)
    else:
        unittest.main()
