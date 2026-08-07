"""E1 feasibility — step 1: baseline replay validation (docs/duet/internal/15 §9 E1).

Contract (design v6): before any counterfactual tree-policy comparison,
the replay model must reproduce the champion's measured decode TPS within
±1% from (a) the E0 trace's per-step status/accepted-length sequence and
(b) a per-status period model fit on the champion profile. If this fails,
counterfactual numbers are not to be trusted (and are not reported).

Inputs:
- E0 trace  : experiments/proxy_async_overlap/e0_collect/run1/e0_draft_*.jsonl
- timing    : experiments/proxy_async_overlap/champion_profile/
              e9k24_jit_profile/duet_profile_target_rank0_*.json
              (per-status step period = diff of successive step anchors)
- reference : final_rematch RESULTS.md — champion decode TPS 81.91
              (mean of 5-rep verdict), tok/step 4.108.

Run:
  cd ssd && python experiments/proxy_async_overlap/e1_feasibility/e1_baseline_replay.py
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E0_DIR = os.path.join(ROOT, "e0_collect", "run1")
PROF = glob.glob(os.path.join(
    ROOT, "champion_profile", "e9k24_jit_profile",
    "duet_profile_target_rank0_*.json"))[0]
REF_TPS = 81.91           # final_rematch verdict mean
REF_TOK_STEP = 4.108      # incl recovery
WARMUP_STEPS = 10

STATUS_NAME = {0: "miss", 1: "hit_k1", 2: "hit_k2"}


def load_period_model():
    """Per-status step period (ms) from the champion target profile:
    period(t) = first-span-start(t+1) − first-span-start(t), attributed to
    the status of step t."""
    recs = json.load(open(PROF))
    start = {}
    status = {}
    for r in recs:
        sid = r.get("step_id")
        if sid is None:
            continue
        s = r.get("start_ms")
        if s is None:
            continue
        if sid not in start or s < start[sid]:
            start[sid] = s
        if r.get("status") in ("miss", "hit_k1", "hit_k2"):
            status[sid] = r["status"]
    per = {}
    for sid in sorted(start):
        nxt = start.get(sid + 1)
        stt = status.get(sid)
        if nxt is None or stt is None or sid <= WARMUP_STEPS:
            continue
        d = nxt - start[sid]
        if 0 < d < 250:                      # 이상치(>250ms) 제외
            per.setdefault(stt, []).append(d)
    return {k: st.mean(v) for k, v in per.items()}, \
           {k: len(v) for k, v in per.items()}


def main():
    period, counts = load_period_model()
    print(f"[period model] (champion profile, warmup 제외, ms/step): "
          f"{ {k: round(v, 2) for k, v in period.items()} } n={counts}")

    # E0 토큰 회계: fan_idx+1 = recovery 포함 커밋 토큰 (docs/duet/internal/17 §3
    # 검증 — 전체 평균 4.086 ≈ 공인 4.108).
    f = glob.glob(os.path.join(E0_DIR, "e0_draft_*.jsonl"))[0]
    req, resp = {}, {}
    for line in open(f):
        r = json.loads(line)
        if r["kind"] == "request":
            c = r["cache_keys"][0]
            req[r["step_id"]] = (c[0], c[1])
        elif r["kind"] == "response":
            resp[r["step_id"]] = r["phase_source"][0]
    tot_ms = 0.0
    tot_tok = 0
    n = 0
    ph_to_status = {0: "miss", 1: "hit_k1", 2: "hit_k2"}
    for t, ph in resp.items():
        o, o0 = req.get(t + 1), req.get(t)
        if o is None or o0 is None or o[0] != o0[0]:
            continue
        stt = ph_to_status[ph]
        tot_ms += period[stt]
        tot_tok += o[1] + 1                  # 커밋 토큰 (recovery 포함)
        n += 1
    tps = tot_tok / (tot_ms / 1000.0)
    print(f"[replay] steps={n} tok/step={tot_tok/n:.3f} "
          f"period(가중)={tot_ms/n:.2f}ms → predicted TPS={tps:.2f}")
    err = (tps - REF_TPS) / REF_TPS
    print(f"[cross-run] 공인 verdict-mean TPS={REF_TPS} → 오차 {err:+.2%} "
          f"(주의: 타이밍은 별도 profile 런에서 적합 — 런-간 주기 편차 포함)")

    # 자기-일관성: 타이밍을 적합한 profile 런 '자신'의 상태 mix + 실측
    # tok/step(4.281 = 41215/9628) + 실측 TPS(82.56, run.log)로 검증 —
    # 모델 오차와 런-간 편차를 분리한다.
    PROF_TPS, PROF_TOKSTEP = 82.56, 41215 / 9628
    mix = {k: counts[k] / sum(counts.values()) for k in counts}
    per_w = sum(period[k] * mix[k] for k in mix)
    tps_self = PROF_TOKSTEP / (per_w / 1000.0)
    err_self = (tps_self - PROF_TPS) / PROF_TPS
    print(f"[self-consistency] profile 런 자신: 모델 주기 {per_w:.2f}ms → "
          f"predicted {tps_self:.2f} vs 실측 {PROF_TPS} = {err_self:+.2%} "
          f"(±1% 합격선의 '모델 정확도' 성분)")


if __name__ == "__main__":
    main()
