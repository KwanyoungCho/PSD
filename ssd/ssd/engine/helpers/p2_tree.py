"""P2-tree helpers — T1 (docs/duet/20; 설계 docs/duet/15 v6).

T1.2: 사전 예산 배분 (결정 ⑤v2 + 외부 리뷰 4차 규약).
무손실 핵심 규약 (D10): 위상/예산 결정은 샘플 정체를 관측하기 **전에**
확정한다 — 이 모듈의 함수들은 자식 토큰 정체를 입력으로 받지 않는다
(시그니처 수준에서 D10 보장).

전부 pure 함수 (텐서 in → 텐서 out, 상태 없음) — CPU 유닛테스트 대상.
"""
import torch


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
