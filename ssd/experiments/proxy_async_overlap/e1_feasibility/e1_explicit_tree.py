#!/usr/bin/env python3
"""이슈 #26 (리뷰 3): E1 상한 재산정 — '명시(explicit) 트리' 기대 수락.

기존 tree_L은 레벨 fanout [2,2,2,2]를 '어느 경로로 가든 대안 f개'로
계산 = 공유/동적 트리 가정 (명시 구현은 2+4+8+16=30노드 필요).
여기서는 실제 topology(par 리스트)의 terminal-mass DP로 기대 수락
길이를 계산한다. 형제 상관은 E1과 동일 규약: k번째 형제의 조건부
수락 a_k = 1-(1-α)^λ (k>=1; 누적 any-accept = 1-(1-α)^{1+λk} 정합).

사용: python e1_explicit_tree.py  (E0 run1 트레이스에서 α 적합)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import e1_arms  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(REPO, "ssd"))
from ssd.engine.helpers.p2_tree import terminal_mass_dp  # noqa: E402
import torch  # noqa: E402

LAM = 0.525          # docs/duet/internal/18: λ=0.51~0.54 (최종층)
D = 4


def node_alpha(depth, sib_order, alpha):
    a = alpha[depth - 1]
    if sib_order == 0:
        return a
    return 1.0 - (1.0 - a) ** LAM      # λ-할인 조건부 (E1 규약 정합)


def expected_L(par, sib, alpha):
    """명시 트리의 기대 수락 길이 (terminal DP × depth)."""
    valid = len(par)
    depth = [0] * valid
    for j in range(valid):
        depth[j] = 1 if par[j] < 0 else depth[par[j]] + 1
    a = torch.tensor([node_alpha(depth[j], sib[j], alpha)
                      for j in range(valid)])
    term = terminal_mass_dp(par, a)
    d_of_ctx = [0] + depth
    return sum(float(term[i]) * d_of_ctx[i] for i in range(valid + 1))


def chain(b):
    par = [-1] + list(range(b - 1))
    return par[:b], [0] * b


def backbone_rescue(budget):
    """backbone 4 + (budget-4) rescue 형제 (얕은 깊이 우선 부착)."""
    par = [-1, 0, 1, 2][:min(4, budget)]
    sib = [0] * len(par)
    extra = budget - len(par)
    attach = 0                          # rescue: 깊이1부터 둘째-형제
    while extra > 0 and attach < len(par):
        anchor_parent = par[attach]     # backbone 노드 attach의 부모
        par.append(anchor_parent)
        sib.append(1)
        extra -= 1
        attach += 1
    return par, sib


def main():
    steps = e1_arms.load_steps()
    alpha, L_obs = e1_arms.fit_alpha(steps)
    print(f"α={[round(x,3) for x in alpha]}  관측 체인 L_p2={L_obs:.3f}")
    pc, sc = chain(4)
    L_chain = expected_L(pc, sc, alpha)
    print(f"\n체인-4 (명시 DP 자기검증): {L_chain:.3f} "
          f"(tree_L(4)={e1_arms.tree_L(4, alpha):.3f})")
    print(f"[구모델] 공유-트리 tree_L(8,[2,2,2,2]) λ=1: "
          f"{e1_arms.tree_L(8, alpha):.3f}  ← 명시 구현 불가(30노드 필요)")
    print(f"[구모델] 동 λ={LAM}: "
          f"{e1_arms.tree_L(8, alpha, lam=LAM):.3f}")
    print("\n명시 backbone+rescue (λ-할인):")
    for b in range(4, 9):
        p, sb = backbone_rescue(b)
        L = expected_L(p, sb, alpha)
        print(f"  budget {b}: L={L:.3f}  (+{(L/L_chain-1)*100:.1f}% vs 체인)")


if __name__ == "__main__":
    main()
