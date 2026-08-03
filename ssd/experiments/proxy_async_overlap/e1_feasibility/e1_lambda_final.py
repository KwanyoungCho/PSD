"""E1 — 형제 상관 λ, **target 최종층 분포** 기반 재계산 (사용자 지적 반영).

exit 근사(e1_lambda_est.py)의 약점을 제거: run3부터 E0가 verify 직후의
logits_p(최종층) top-32 + 정확 lse를 kind="final"로 기록한다. k번째
"final"은 같은 step의 k번째 "wire"(draft top-32 보유)와 짝.

검증 내장: 최종 분포라면 형제-1 예측 Σmin(p,q)가 실측 α와 보정 없이
맞아야 한다 — 이 일치도가 방법 전체의 신뢰도 지표.

Run: cd ssd && python experiments/proxy_async_overlap/e1_feasibility/e1_lambda_final.py
"""
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E0 = os.path.join(ROOT, "e0_collect", "run3_final")
ALPHA_ACT = [0.675, 0.709, 0.751, 0.785]     # 실측 (run1 적합; run3 재적합 병기)
MAX_STEPS = 4000


def softmax_t(ids, logits, T):
    m = max(logits)
    ex = [math.exp((x - m) / max(T, 1e-6)) for x in logits]
    Z = sum(ex)
    return {i: e / Z for i, e in zip(ids, ex)}


def ladder_pair(p, q):
    supp = set(p) | set(q)
    E1 = sum(min(p.get(v, 0.0), q.get(v, 0.0)) for v in supp)
    resid = {v: max(0.0, p.get(v, 0.0) - q.get(v, 0.0)) for v in supp}
    Z = sum(resid.values())
    if Z <= 1e-12:
        return E1, 0.0
    pp = {v: r / Z for v, r in resid.items()}
    rej_tot = acc2_tot = 0.0
    for v in q:
        w_rej = max(0.0, q[v] - p.get(v, 0.0))
        if w_rej <= 0.0:
            continue
        d2n = 1.0 - q[v]
        if d2n <= 1e-9:
            continue
        a2 = sum(min(q[w] / d2n, pp.get(w, 0.0)) for w in q if w != v)
        rej_tot += w_rej
        acc2_tot += w_rej * a2
    return E1, (acc2_tot / rej_tot if rej_tot > 0 else 0.0)


def main():
    f = glob.glob(os.path.join(E0, "e0_target_*.jsonl"))[0]
    wires, finals = [], []
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") == "wire":
            wires.append(r)
        elif r.get("kind") == "final":
            finals.append(r)
    n_pair = min(len(wires), len(finals))
    print(f"[data] wire {len(wires)} / final {len(finals)} → 짝 {n_pair}")
    sums = [[0.0, 0.0, 0] for _ in range(4)]
    kmis = 0
    used = 0
    for w, fi in zip(wires[:n_pair], finals[:n_pair]):
        if w["K"] != fi["K"]:
            kmis += 1
            continue
        if w["K"] != 4:
            continue
        T = fi["temps"][0] if fi.get("temps") else 0.7
        dids, dlg = w["draft_top_ids"][0], w["draft_top_logits"][0]
        pids, plg = fi["final_top_ids"][0], fi["final_top_logits"][0]
        for d in range(4):
            p = softmax_t(pids[d], plg[d], T)     # 최종층, target temp
            q = softmax_t(dids[d], dlg[d], T)     # draft, 같은 temp
            E1, E2r = ladder_pair(p, q)
            sums[d][0] += E1
            sums[d][1] += E2r
            sums[d][2] += 1
        used += 1
        if used >= MAX_STEPS:
            break
    print(f"[data] K=4 짝 사용 {used} (K 불일치 {kmis} — 0이어야 정상)")
    lams = []
    for d in range(4):
        E1, E2r, nn = sums[d]
        E1 /= nn
        E2r /= nn
        a = ALPHA_ACT[d]
        # 검증: 최종 분포면 E1 ≈ 실측 α여야 함 (보정 불필요가 이상적)
        c = a / E1
        a2r_raw = E2r
        a2r_cal = min(0.999, c * E2r)
        lam_raw = math.log(max(1e-9, 1 - a2r_raw)) / math.log(1 - a)
        lam_cal = math.log(max(1e-9, 1 - a2r_cal)) / math.log(1 - a)
        lams.append((lam_raw, lam_cal))
        print(f"  레벨 {d}: E1(최종)={E1:.3f} vs 실측 α={a:.3f} "
              f"(비율 {c:.2f} — 1.0에 가까울수록 방법 신뢰) | "
              f"a₂|기각={E2r:.3f} → λ raw={lam_raw:.2f} / 보정={lam_cal:.2f}")
    mr = sum(x for x, _ in lams) / 4
    mc = sum(y for _, y in lams) / 4
    print(f"\n[결론 입력] λ̂(최종층) raw={mr:.2f} / 보정={mc:.2f} "
          f"(exit-근사 추정치 0.53과 비교)")
    print("남은 한계: top-32 절단, sampler_x 미반영.")


if __name__ == "__main__":
    main()
