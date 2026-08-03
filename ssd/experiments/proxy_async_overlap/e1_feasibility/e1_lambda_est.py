"""E1 — 형제 상관 λ 추정 (경로 i: E0 기록 분포의 사다리 계산).

Step 3-B가 보인 결정적 불확실성: 트리 이득은 "맏이 기각 시 형제-2가
수락될 확률 a₂r"에 걸려 있다 (λ = ln(1−a₂r)/ln(1−α); λ=1 독립, 0 무가치).

방법 (근사 명시):
- E0 wire 레코드의 위치별 top-32 exit logits(+정확한 lse@temp1)와
  top-32 draft logits로 p̂(exit), q(draft) 분포를 복원. temperature
  0.7 적용은 top-32 자기-정규화 (T<1 첨예화로 top-32 질량 지배 —
  근사 오차 작음; sampler_x 미적용은 한계로 명시).
- 사다리 (§7.2 정확 재귀): P(s1=v ∧ 기각) = max(0, q(v)−p(v)),
  잔차 p' = norm((p−q)₊), D₂ = q\\{v} 재정규화,
  a₂|s1=v = Σ_w min(D₂(w), p'(w)). 꼬리 질량은 스칼라로 보수 반영.
- **보정**: 같은 추정기의 형제-1 예측 Σmin(p,q)를 실측 α_d(레벨별,
  P2-hit 적합)와 대조 → c_d = α_d / E1_est_d 를 형제-2 추정에도 적용
  (같은 곱셈 편향 가정 — 한계로 명시. exit≠최종p 편향의 1차 보정).
- K=4 (P2/miss 응답) step의 위치 0..3만 사용 — 트리가 사는 레짐.

Run: cd ssd && python experiments/proxy_async_overlap/e1_feasibility/e1_lambda_est.py
"""
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E0 = os.path.join(ROOT, "e0_collect", "run1")
TEMP = 0.7
ALPHA_ACT = [0.675, 0.709, 0.751, 0.785]      # 실측 (e1_arms 적합)
MAX_STEPS = 4000                               # K=4 wire 전수(946*?) 충분


def softmax_t(ids, logits, T):
    m = max(logits)
    ex = [math.exp((x - m) / T) for x in logits]
    Z = sum(ex)
    return {i: e / Z for i, e in zip(ids, ex)}


def ladder_pair(p, q):
    """(E1_est, E2r_est): 형제-1 기대 수락률, 형제-2 조건부 수락률.
    p, q: dict token->prob (top-32 자기-정규화)."""
    supp = set(p) | set(q)
    overlap = sum(min(p.get(v, 0.0), q.get(v, 0.0)) for v in supp)
    E1 = overlap                                  # Σ min(p,q)
    # 잔차 p' = norm((p−q)+)
    resid = {v: max(0.0, p.get(v, 0.0) - q.get(v, 0.0)) for v in supp}
    Z = sum(resid.values())
    rej_tot = 0.0
    acc2_tot = 0.0
    if Z <= 1e-12:
        return E1, 0.0
    pp = {v: r / Z for v, r in resid.items()}
    for v in q:                                   # s1 = v 가 기각되는 경우
        w_rej = max(0.0, q[v] - p.get(v, 0.0))
        if w_rej <= 0.0:
            continue
        d2_norm = 1.0 - q[v]
        if d2_norm <= 1e-9:
            continue
        a2 = sum(min(q[w] / d2_norm, pp.get(w, 0.0))
                 for w in q if w != v)
        rej_tot += w_rej
        acc2_tot += w_rej * a2
    E2r = acc2_tot / rej_tot if rej_tot > 0 else 0.0
    return E1, E2r


def main():
    f = glob.glob(os.path.join(E0, "e0_target_*.jsonl"))[0]
    sums = [[0.0, 0.0, 0] for _ in range(4)]      # level -> [E1, E2r, n]
    used = 0
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") != "wire" or r["K"] != 4:
            continue
        eids, elg = r["exit_top_ids"][0], r["exit_top_logits"][0]
        dids, dlg = r["draft_top_ids"][0], r["draft_top_logits"][0]
        for d in range(4):                        # 위치 0..3 = 레벨
            p = softmax_t(eids[d], elg[d], TEMP)
            q = softmax_t(dids[d], dlg[d], TEMP)
            E1, E2r = ladder_pair(p, q)
            sums[d][0] += E1
            sums[d][1] += E2r
            sums[d][2] += 1
        used += 1
        if used >= MAX_STEPS:
            break
    print(f"[data] K=4 wire steps 사용: {used}")
    lam_list = []
    for d in range(4):
        E1, E2r, nn = sums[d]
        E1 /= nn
        E2r /= nn
        a = ALPHA_ACT[d]
        c = a / E1                                # 보정 (exit≈p 편향)
        a2r = min(0.999, c * E2r)
        lam = math.log(max(1e-9, 1 - a2r)) / math.log(1 - a)
        lam_list.append(lam)
        print(f"  레벨 {d}: E1_est={E1:.3f} (실측 α={a:.3f}, 보정 c={c:.2f}) "
              f"| E2r_est={E2r:.3f} → 보정 후 a₂|기각={a2r:.3f} "
              f"| λ̂={lam:.2f}")
    lam_mean = sum(lam_list) / len(lam_list)
    print(f"\n[결론 입력] λ̂ 평균 = {lam_mean:.2f} "
          f"(레벨별 {[round(x, 2) for x in lam_list]})")
    print("한계: exit≈최종p 근사(1차 보정만), top-32 절단, sampler_x "
          "미반영 — 확정은 HF 소표본 재생(경로 ii) 또는 E2 실측.")


if __name__ == "__main__":
    main()
