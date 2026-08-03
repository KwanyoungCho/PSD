"""P2-tree helpers — T1 (docs/duet/20; 설계 docs/duet/15 v6).

T1.2: 사전 예산 배분 (결정 ⑤v2 + 외부 리뷰 4차 규약).
무손실 핵심 규약 (D10): 위상/예산 결정은 샘플 정체를 관측하기 **전에**
확정한다 — 이 모듈의 함수들은 자식 토큰 정체를 입력으로 받지 않는다
(시그니처 수준에서 D10 보장).

전부 pure 함수 (텐서 in → 텐서 out, 상태 없음) — CPU 유닛테스트 대상.
"""
import torch

from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling


def alloc_root_budgets(piv: torch.Tensor, total: int, beta: float,
                       cap: int) -> torch.Tensor:
    """root별 자식 예산 b_root ∝ P_iv^β (결정 ⑤v2; β=0.5 — E1 근거).

    Args:
        piv:   [R] 각 root의 라이브 P_iv (wire 수신값; draw-전 정보).
        total: 총 노드 예산 (= F_total × W).
        beta:  배분 지수 (0=균등, 1=비례).
        cap:   root당 상한 (= N_v — 응답에 못 들어갈 초과 생성 방지).

    Returns: [R] int64, sum ≤ total, 각 ≤ cap. 결정론적 (largest-
    remainder 배분; 동률은 낮은 인덱스 = 높은 rank 우선).
    """
    R = piv.numel()
    w = piv.clamp_min(1e-9).double().pow(beta)
    quota = w / w.sum() * total                       # 실수 몫
    base = quota.floor().long().clamp_max(cap)
    rem_budget = int(total - base.sum().item())
    if rem_budget > 0:
        frac = quota - quota.floor()
        # cap 도달 root는 추가 배분 제외
        frac = torch.where(base >= cap, torch.full_like(frac, -1.0), frac)
        order = torch.argsort(frac, descending=True, stable=True)
        for i in order.tolist():
            if rem_budget <= 0:
                break
            if base[i] < cap:
                base[i] += 1
                rem_budget -= 1
    return base


def alloc_fanouts(parent_priority: torch.Tensor,
                  parent_root: torch.Tensor,
                  root_remaining: torch.Tensor,
                  c_tensor: int) -> torch.Tensor:
    """같은 forward에서 평가되는 부모들의 fanout b_x (draw-전 확정).

    외부 리뷰 4차 규약: 같은 root의 부모들이 remaining을 동시에 읽으면
    cap 초과 — priority 내림차순(동률: 낮은 인덱스) 정렬 후 root별
    누적 prefix 배분으로 결정론화.

    Args:
        parent_priority: [W] 부모 priority (log 공간; 이미 관측된 정보).
        parent_root:     [W] 부모의 root 인덱스.
        root_remaining:  [R] root별 잔여 자식 예산 (호출자가 갱신 관리).
        c_tensor:        노드당 샘플 상한.

    Returns: [W] int64 fanout (0 허용 — 예산 소진 부모).
    """
    W = parent_priority.numel()
    out = torch.zeros(W, dtype=torch.int64)
    remaining = root_remaining.clone()
    order = torch.argsort(parent_priority, descending=True, stable=True)
    for i in order.tolist():
        r = int(parent_root[i])
        take = min(c_tensor, int(remaining[r]))
        if take > 0:
            out[i] = take
            remaining[r] -= take
    return out


def tree_sample_wor(logits: torch.Tensor, temperatures: torch.Tensor,
                    c_tensor: int, sampler_x=None, F=None):
    """비복원(WOR) C_tensor개 샘플 — 순서 보존 (T1.3; D8/D11).

    구현: exponential-race top-k — race 점수 내림차순 = 순차 비복원
    추출 순서 (지수시계 표준 성질). **c_tensor=1이면 현행
    Sampler.forward와 op 시퀀스·RNG 소비·결과가 bit-identical**
    (fast-path 게이트의 근거; 테스트로 고정).

    temp==0 금지 (v6 게이트: 트리 OFF 폴백은 호출자 책임) — one-hot
    support에서 2번째 비복원 추출이 미정의이기 때문 (fallback 미구현
    을 의도적으로 게이트).

    Returns:
        tokens [B, C] int64 — WOR 순서 (형제 순서 기록 그 자체).
        raw_q  [B, C] float — **원본 q_eff 확률** (재정규화 전 —
            결정 ② c_raw 규약: priority는 이 값으로 계산).
    """
    if bool((temperatures <= 0).any()):
        raise ValueError(
            "tree_sample_wor: temperature==0 is gated (v6 §7.2 — "
            "support-exhaustion fallback intentionally not implemented; "
            "caller must fall back to the chain path)")
    logits_cpy = logits.to(torch.float)
    logits_cpy.div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits_cpy, dim=-1, dtype=torch.float)
    if sampler_x is not None:
        probs = apply_sampler_x_rescaling(probs, sampler_x, F)
    raw_q = probs.clone()                      # 원본 보존 (c_raw)
    epsilon = 1e-10
    scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
    if c_tensor == 1:
        tokens = scores.argmax(dim=-1, keepdim=True)   # Sampler와 동일 op
    else:
        tokens = scores.topk(c_tensor, dim=-1).indices
    return tokens, raw_q.gather(1, tokens)
