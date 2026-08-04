"""P2-tree helpers — T1 (docs/duet/20; 설계 docs/duet/15 v6).

T1.2: 사전 예산 배분 (결정 ⑤v2 + 외부 리뷰 4차 규약).
무손실 핵심 규약 (D10): 위상/예산 결정은 샘플 정체를 관측하기 **전에**
확정한다 — 이 모듈의 함수들은 자식 토큰 정체를 입력으로 받지 않는다
(시그니처 수준에서 D10 보장).

전부 pure 함수 (텐서 in → 텐서 out, 상태 없음) — CPU 유닛테스트 대상.
"""
import numpy as np
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
        # 이슈 #23 (리뷰 2A): capped water-filling — 종전 단일-패스
        # (+1은 root당 1회)는 cap-절단 질량을 버려 β=1·cap=8에서 평균
        # 32/40만 소진 (최저 18). frac 내림차순 라운드-로빈으로 미달
        # root들에 소진될 때까지 재배분: sum == min(total, R·cap) 보장.
        frac = quota - quota.floor()
        order = torch.argsort(frac, descending=True, stable=True).tolist()
        progressed = True
        while rem_budget > 0 and progressed:
            progressed = False
            for i in order:
                if rem_budget <= 0:
                    break
                if base[i] < cap:
                    base[i] += 1
                    rem_budget -= 1
                    progressed = True
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


def alloc_fanouts_backbone(parent_priority: torch.Tensor,
                           parent_root: torch.Tensor,
                           root_remaining: torch.Tensor,
                           root_reserve: torch.Tensor,
                           is_tip: torch.Tensor,
                           c_tensor: int) -> torch.Tensor:
    """backbone-우선 fanout (형상 진단 2026-08-04 — docs/duet/21 §4.5).

    고정-C 배분은 예산을 폭으로 소진해 깊이가 죽는다 (C=1은 형제 0,
    C≥2는 dmax≈K2/2 — E1 승리 형상 '깊이 유지+형제 추가'가 생성
    불가). 규칙:
      1) 백본 tip(각 root의 맏이-사슬 끝, 깊이 미완)은 fan 1을 최우선
         보장 — 모든 root가 최소한 체인-동형 기저를 가진다.
      2) 형제 추가는 (remaining - reserve)의 잔여 예산에서만 priority
         내림차순 +1씩 (노드당 c_tensor 상한). reserve = 백본 완성까지
         남은 깊이 — 미래의 백본 연장분을 형제가 먹지 못하게 예약.

    Args:
        root_reserve: [R] 백본 완성까지 남은 깊이 (tip fan 1 배정 시
            호출자가 -1; 여기서는 읽기만).
        is_tip: [W] bool — 해당 부모가 자기 root의 백본 tip인가.

    Returns: [W] int64 fanout.
    """
    W = parent_priority.numel()
    out = torch.zeros(W, dtype=torch.int64)
    remaining = root_remaining.clone()
    reserve = root_reserve.clone()
    # 1) 백본 tip: fan 1 보장 (예약분 소비)
    for i in range(W):
        if bool(is_tip[i]):
            r = int(parent_root[i])
            if int(remaining[r]) > 0:
                out[i] = 1
                remaining[r] -= 1
                reserve[r] = max(0, int(reserve[r]) - 1)
    # 2) 잔여-예약 차감분에서 priority-prefix로 형제 +1씩
    order = torch.argsort(parent_priority, descending=True, stable=True)
    progressed = True
    while progressed:
        progressed = False
        for i in order.tolist():
            r = int(parent_root[i])
            avail = int(remaining[r]) - int(reserve[r])
            if avail > 0 and int(out[i]) < c_tensor:
                out[i] += 1
                remaining[r] -= 1
                progressed = True
    return out


def q_probs_from_logits(logits: torch.Tensor, temperatures: torch.Tensor,
                        sampler_x=None, F=None):
    """draft 제안분포 q 빌드 — 샘플측(tree_sample_wor)과 verify측
    (T3.4-b5 보행의 q_parent_probs)이 **동일 함수**를 쓴다 (수락 보존의
    전제: 같은 logits → 같은 q). op 시퀀스는 기존 tree_sample_wor
    인라인과 bit-identical (c=1 RNG-parity 테스트가 고정).
    """
    logits_cpy = logits.to(torch.float)
    logits_cpy.div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits_cpy, dim=-1, dtype=torch.float)
    if sampler_x is not None:
        probs = apply_sampler_x_rescaling(probs, sampler_x, F)
    return probs


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
    probs = q_probs_from_logits(logits, temperatures, sampler_x, F)
    raw_q = probs.clone()                      # 원본 보존 (c_raw)
    epsilon = 1e-10
    scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
    if c_tensor == 1:
        tokens = scores.argmax(dim=-1, keepdim=True)   # Sampler와 동일 op
    else:
        tokens = scores.topk(c_tensor, dim=-1).indices
    return tokens, raw_q.gather(1, tokens)


# --- T2.0: P_iv wire 비트-pack (설계 v6 D2; tree policy != off 게이트) ---
PIV_SHIFT = 15                      # 토큰은 비트 0-14 (vocab ≤ 32768 가드)
PIV_BITS = 16                       # log10 P_iv ∈ [-6, 0] 16비트 양자화
PIV_VER_BIT = 31                    # 버전/유효 마커
_PIV_QMAX = (1 << PIV_BITS) - 1
_TOK_MASK = (1 << PIV_SHIFT) - 1
_LOG_MIN = -6.0


def pack_piv(chosen_tok: torch.Tensor, piv: torch.Tensor) -> torch.Tensor:
    """chosen_tok int64의 비트 15-30에 양자화 log10(P_iv)를 pack.

    양자화 오차 ≤ 반스텝 (6데케이드/65535 ≈ 9.2e-5 데케이드). NCCL
    호출 수·크기 불변 (같은 int64 자리). 수신측은 dedup **이전**에
    unpack해야 한다 (D2 정확성 함정 — draft_runner에서 보장)."""
    q = ((piv.clamp_min(1e-9).log10().clamp(_LOG_MIN, 0.0) - _LOG_MIN)
         / (-_LOG_MIN) * _PIV_QMAX).round().long()
    return chosen_tok | (q << PIV_SHIFT) | (1 << PIV_VER_BIT)


def unpack_piv(packed: torch.Tensor):
    """→ (clean_tok [.,N] int64, piv [.,N] float32). 버전 비트 검증."""
    if not bool((packed >> PIV_VER_BIT & 1).all()):
        raise ValueError("unpack_piv: version bit missing — "
                         "pack 게이트 불일치 (tree policy 양단 확인)")
    tok = packed & _TOK_MASK
    q = (packed >> PIV_SHIFT) & _PIV_QMAX
    piv = torch.pow(10.0, q.float() / _PIV_QMAX * (-_LOG_MIN) + _LOG_MIN)
    return tok, piv


# --- T1.4a: rollout 알고리즘 골격 (pure — 엔진 배선은 T1.4b) ---

class TreePool:
    """고정 용량 pool (설계 v6 §6 실행 모델). CPU/GPU 텐서 필드.

    상태: 0=미평가(candidate, 확장 가능), 1=평가완료(D11 — 재방문 없음).
    모든 노드는 생성 즉시 candidate tree의 일원 (D10 — 잎 포함).
    """

    def __init__(self, capacity: int, device="cpu"):
        d = device
        self.tok = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.parent_cell = torch.full((capacity,), -1, dtype=torch.int64,
                                      device=d)   # -1 = root (부모=글루)
        self.parent_idx = torch.full((capacity,), -1, dtype=torch.int64,
                                     device=d)    # pool 인덱스 (조상 복원)
        self.depth = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.root = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.sib_order = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.logpri = torch.full((capacity,), float("-inf"), device=d)
        self.raw_q = torch.ones(capacity, device=d)   # c_raw (원본 확률)
        self.state = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.cell = torch.full((capacity,), -1, dtype=torch.int64, device=d)
        self.n = 0

    def add(self, tok, parent_idx, parent_cell, depth, root, sib, logpri,
            raw_q):
        i = self.n
        self.tok[i] = tok
        self.parent_idx[i] = parent_idx
        self.parent_cell[i] = parent_cell
        self.depth[i] = depth
        self.root[i] = root
        self.sib_order[i] = sib
        self.logpri[i] = logpri
        self.raw_q[i] = raw_q
        self.n += 1
        return i

    def ancestors_cells(self, i):
        """노드 i의 조상 셀 목록 (mask 비트용; root 자신 제외 글루까지)."""
        cells = []
        j = int(self.parent_idx[i])
        while j >= 0:
            if int(self.cell[j]) >= 0:
                cells.append(int(self.cell[j]))
            j = int(self.parent_idx[j])
        return cells


def select_nodes(pool: TreePool, policy: str, W: int, fwd: int,
                 depth_cap: int):
    """이번 forward에서 평가할 노드 선택 (D1 정책 스위치).

    level    : depth == fwd 인 미평가 노드만 (level-synchronous).
    frontier : 미평가 전체에서 priority 상위 W.
    공통: depth ≥ depth_cap 노드는 제외 (자식이 캡 초과라 확장 무의미).
    반환: pool 인덱스 리스트 (≤ W; 부족하면 그만큼 — pad는 호출자).
    """
    n = pool.n
    elig = (pool.state[:n] == 0) & (pool.depth[:n] < depth_cap)
    if policy == "level":
        elig = elig & (pool.depth[:n] == fwd)
    idx = torch.nonzero(elig).flatten()
    if idx.numel() == 0:
        return []
    pri = pool.logpri[idx]
    order = torch.argsort(pri, descending=True, stable=True)
    return idx[order][:W].tolist()


def rollout_reference(root_toks, root_piv, root_pos, *, policy, W, F_total,
                      c_tensor, nv, beta, depth_cap, sample_fn,
                      fanout_policy="ctensor"):
    """rollout 참조 구현 (T1.4a — 엔진 배선(T1.4b)의 정답지이자
    CPU 테스트 대상). sample_fn(node_indices, fanouts) -> (tokens [n, C],
    raw_q [n, C]) — 정체는 여기서만 관측 (D10: 예산·선택은 그 전에 확정).

    반환: (pool, eval_log) — eval_log[f] = (선택 인덱스, fanout) 기록.
    """
    R = len(root_toks)
    pool = TreePool(capacity=R + F_total * W * c_tensor)
    budgets = alloc_root_budgets(root_piv, total=F_total * W, beta=beta,
                                 cap=nv)
    remaining = budgets.clone()
    logpiv = root_piv.clamp_min(1e-9).log()
    for r in range(R):
        pool.add(root_toks[r], -1, -1, 0, r, 0, float(logpiv[r]), 1.0)
    eval_log = []
    tip_idx = list(range(R))                      # backbone tip (root부터)
    for f in range(F_total):
        sel = select_nodes(pool, policy, W, f, depth_cap)
        if not sel:
            eval_log.append(([], None))
            continue
        pri = pool.logpri[torch.tensor(sel)]
        roots = pool.root[torch.tensor(sel)]
        if fanout_policy == "backbone":
            reserve = torch.tensor(
                [max(0, depth_cap - int(pool.depth[tip_idx[r]]))
                 for r in range(R)], dtype=torch.int64)
            is_tip = torch.tensor(
                [sel[k] == tip_idx[int(roots[k])] for k in range(len(sel))],
                dtype=torch.bool)
            fan = alloc_fanouts_backbone(pri, roots, remaining, reserve,
                                         is_tip, c_tensor)
        else:
            fan = alloc_fanouts(pri, roots, remaining, c_tensor)
        # 예산 소진 반영 (draw 전 확정 — D10)
        for k, i in enumerate(sel):
            remaining[int(roots[k])] -= int(fan[k])
        toks, raws = sample_fn(sel, fan)          # 정체 관측은 여기부터
        for k, i in enumerate(sel):
            cell = f * W + k                      # 셀 주소 규칙 (§6 v6)
            pool.cell[i] = cell
            pool.state[i] = 1                     # D11: single-shot
            for c in range(int(fan[k])):
                child = pool.add(int(toks[k][c]), i, cell,
                         int(pool.depth[i]) + 1, int(pool.root[i]), c,
                         float(pool.logpri[i])
                         + float(torch.log(torch.clamp(raws[k][c], min=1e-9))),
                         float(raws[k][c]))
                # backbone 연장: tip의 맏이(c=0)가 새 tip
                if fanout_policy == "backbone" and c == 0 \
                        and i == tip_idx[int(pool.root[i])]:
                    tip_idx[int(pool.root[i])] = child
        eval_log.append((sel, fan))
    return pool, eval_log


def build_tree_mask_packed(fwd, W, K_glue, context_len, prefix_glue_rows,
                           ancestor_cells, self_cols):
    """forward `fwd`의 packed attention mask (T1.4b; 기존 chain 빌더의
    기하를 정확히 복제 — cudagraph_helpers cpu_packed_masks와 동일 규약:
    [prefix 1s | glue (K_glue+1) | spec 블록 (fwd+1)개 × W], packbits
    little, B=1 세그먼트).

    Args:
        fwd:          현재 forward 인덱스 (0-base).
        W:            행 수 (= MQ_LEN).
        K_glue:       글루 폭-1 (= K_for_mask; split_k2는 K2).
        context_len:  이 seq의 context_lens (chain 빌더의 cols 산식 입력).
        prefix_glue_rows: [W, K_glue+1] uint8 — 행별 글루 가시성 (root의
            원 seed-행 글루 패턴을 선택 순서로 재배열한 것).
        ancestor_cells: 행별 조상 셀 목록 (cell = f'·W + k').
        self_cols:    [W] 각 행의 자기 셀 열 (보통 fwd·W + k; pad 행 -1).

    Returns: (packed uint8 np.ndarray, indptr int32 np.ndarray)
    """
    cols = int(context_len) + fwd * W
    ttl_added = (fwd + 1) * W + (K_glue + 1)
    prefix_len = cols - ttl_added
    m = np.zeros((W, cols), dtype=np.uint8)
    m[:, :prefix_len] = 1
    m[:, prefix_len:prefix_len + K_glue + 1] = prefix_glue_rows
    spec0 = prefix_len + K_glue + 1
    for k in range(W):
        for c in ancestor_cells[k]:
            m[k, spec0 + c] = 1
        if self_cols[k] >= 0:
            m[k, spec0 + int(self_cols[k])] = 1
    packed = np.packbits(m.ravel(), bitorder="little")
    indptr = np.array([0, len(packed)], dtype=np.int32)
    return packed, indptr


def run_rollout(root_toks, root_piv, *, policy, W, F_total, c_tensor, nv,
                beta, depth_cap, temps, forward_fn, glue_rows_by_root,
                rope_base_by_root, K_glue, context_len, sampler_x=None,
                F_x=None, pad_token=0, fanout_policy="ctensor"):
    """엔진-주입형 rollout (T1.4b-a): forward_fn만 바꾸면 stub/실엔진
    양쪽에서 동작. per-forward로 (input_ids[W], rope[W], packed mask)를
    구성해 forward_fn(f, input_ids, rope, packed, indptr) -> logits[W,V]
    를 호출하고, tree_sample_wor로 자식을 뽑아 pool을 채운다.

    rollout_reference와 topology가 동일해야 한다 (stub 테스트로 고정).
    pad 행: 유효 노드가 W 미만이면 pad_token/rope_base[0]로 채우고
    fanout 0 (자식 무시; RNG는 소비 — 고정 shape 유지).
    """
    R = len(root_toks)
    pool = TreePool(capacity=R + F_total * W * c_tensor)
    budgets = alloc_root_budgets(root_piv, total=F_total * W, beta=beta,
                                 cap=nv)
    remaining = budgets.clone()
    logpiv = root_piv.clamp_min(1e-9).log()
    for r in range(R):
        pool.add(int(root_toks[r]), -1, -1, 0, r, 0, float(logpiv[r]), 1.0)
    eval_log = []
    cell_logits = None                     # [F·W, V] — verify q_eff 원천 (T2.2)
    tip_idx = list(range(R))               # backbone tip (root 자신부터)
    for f in range(F_total):
        sel = select_nodes(pool, policy, W, f, depth_cap)
        n_sel = len(sel)
        fan = torch.zeros(W, dtype=torch.int64)
        if n_sel:
            pri = pool.logpri[torch.tensor(sel)]
            roots = pool.root[torch.tensor(sel)]
            if fanout_policy == "backbone":
                reserve = torch.tensor(
                    [max(0, depth_cap - int(pool.depth[tip_idx[r]]))
                     for r in range(R)], dtype=torch.int64)
                is_tip = torch.tensor(
                    [sel[k] == tip_idx[int(roots[k])]
                     for k in range(n_sel)], dtype=torch.bool)
                fan[:n_sel] = alloc_fanouts_backbone(
                    pri, roots, remaining, reserve, is_tip, c_tensor)
            else:
                fan[:n_sel] = alloc_fanouts(pri, roots, remaining, c_tensor)
            for k, i in enumerate(sel):
                remaining[int(roots[k])] -= int(fan[k])
        # --- per-forward 텐서 구성 (동적 3요소) ---
        input_ids = torch.full((W,), pad_token, dtype=torch.int64)
        rope = torch.full((W,), int(rope_base_by_root[0]),
                          dtype=torch.int64)
        glue = np.zeros((W, K_glue + 1), dtype=np.uint8)
        anc = [[] for _ in range(W)]
        selfc = [-1] * W
        for k, i in enumerate(sel):
            r = int(pool.root[i])
            input_ids[k] = pool.tok[i]
            rope[k] = int(rope_base_by_root[r]) + int(pool.depth[i])
            glue[k] = glue_rows_by_root[r]
            anc[k] = pool.ancestors_cells(i)
            selfc[k] = f * W + k
        packed, indptr = build_tree_mask_packed(
            f, W, K_glue, context_len, glue, anc, selfc)
        logits = forward_fn(f, input_ids, rope, packed, indptr)
        if cell_logits is None:
            # forward_fn 디바이스 상주 (엔진=GPU — [W,V] CPU 왕복 제거;
            # stub 테스트는 CPU 그대로)
            cell_logits = torch.zeros(F_total * W, logits.shape[-1],
                                      dtype=logits.dtype,
                                      device=logits.device)
        cell_logits[f * W:(f + 1) * W] = logits[:W]
        toks, raws = tree_sample_wor(logits, temps.to(logits.device),
                                     c_tensor, sampler_x=sampler_x, F=F_x)
        toks, raws = toks.cpu(), raws.cpu()   # pool 장부는 CPU (소량 1회)
        for k, i in enumerate(sel):
            cell = f * W + k
            pool.cell[i] = cell
            pool.state[i] = 1
            for c in range(int(fan[k])):
                child = pool.add(int(toks[k][c]), i, cell,
                         int(pool.depth[i]) + 1, int(pool.root[i]), c,
                         float(pool.logpri[i])
                         + float(torch.log(torch.clamp(raws[k][c],
                                                       min=1e-9))),
                         float(raws[k][c]))
                if fanout_policy == "backbone" and c == 0 \
                        and i == tip_idx[int(pool.root[i])]:
                    tip_idx[int(pool.root[i])] = child
        eval_log.append((sel, fan[:n_sel]))
    return pool, eval_log, cell_logits


def build_root_views(pool: TreePool, R: int, nv: int, cell_logits=None):
    """root별 서브트리 응답 뷰 (T1.5; U_max=N_v 고정 pad — v6 §7.1).

    결정 ⑤v2의 생성-시점 캡(≤ nv) 덕분에 **절단이 발생하지 않는다**
    (검증 assert). 뷰 노드 순서 = 생성 순서(= 셀 순서와 일치 — 부모가
    항상 자식보다 앞) → parent_local이 항상 자기보다 앞 (verify 보행
    invariant, 리뷰4 row/slot 규약).

    Returns dict of [R, nv] 텐서: tok / parent_local(-1=root직결) /
    sib_order / raw_q / valid([R] 유효 노드 수).
    """
    tok = torch.zeros(R, nv, dtype=torch.int64)
    parent_local = torch.full((R, nv), -1, dtype=torch.int64)
    sib = torch.zeros(R, nv, dtype=torch.int64)
    raw_q = torch.zeros(R, nv)
    valid = torch.zeros(R, dtype=torch.int64)
    local_of = {}
    for i in range(pool.n):
        p = int(pool.parent_idx[i])
        if p < 0:
            continue                       # root 자체는 뷰에 안 들어감
        r = int(pool.root[i])
        j = int(valid[r])
        assert j < nv, "생성-시점 캡 위반 (⑤v2 — 있을 수 없음)"
        tok[r, j] = pool.tok[i]
        pp = int(pool.parent_idx[i])
        parent_local[r, j] = local_of.get(pp, -1)   # 부모가 root면 -1
        sib[r, j] = pool.sib_order[i]
        raw_q[r, j] = pool.raw_q[i]
        local_of[i] = j
        valid[r] = j + 1
    out = {"tok": tok, "parent_local": parent_local, "sib_order": sib,
           "raw_q": raw_q, "valid": valid}
    if cell_logits is not None:
        # T2.2: parent_q 참조 — root별 고유 부모 셀 (첫 등장 순), U ≤ nv.
        V = cell_logits.shape[-1]
        pq_ref = torch.full((R, nv), -1, dtype=torch.int64)
        pq_logits = torch.zeros(R, nv, V, dtype=cell_logits.dtype,
                                device=cell_logits.device)
        u_valid = torch.zeros(R, dtype=torch.int64)
        uniq = [dict() for _ in range(R)]
        # 다시 순회하며 노드별 부모 셀 → 로컬 U 인덱스
        cnt = [0] * R
        for i in range(pool.n):
            if int(pool.parent_idx[i]) < 0:
                continue
            r = int(pool.root[i])
            pc = int(pool.parent_cell[i])
            if pc not in uniq[r]:
                u = len(uniq[r])
                assert u < nv, "U_max=N_v 위반"
                uniq[r][pc] = u
                pq_logits[r, u] = cell_logits[pc]
                u_valid[r] = u + 1
            pq_ref[r, cnt[r]] = uniq[r][pc]
            cnt[r] += 1
        out["parent_q_ref"] = pq_ref
        out["parent_q_logits"] = pq_logits
        out["u_valid"] = u_valid
    return out


def tree_verify_walk(view, p_dists, q_dists, root_p, rng):
    """무손실 트리 수락 보행 (T3.3 — v6 §7.2 + 리뷰4 수치 규약).

    참조 구현 (CPU, dict 분포): 전수 분포-일치 테스트의 대상이며 T3
    엔진 보행의 정답지. 형제 그룹별 잔차 사다리:
        R₁=p, D₁=q(재정규화 없음 — 원본), a_j = min(1, R_j[x]/D_j[x]),
        R_{j+1} = norm((R_j − D_j)₊), D_{j+1} = x_j 제거·재정규화.
    내부 수락 → 자식 그룹은 **새 컨텍스트 p로 리셋**. 수락된 잎 →
    plain p에서 bonus. 전원 기각 → 마지막 잔차에서 recovery.

    Args:
        view: build_root_views의 한 root 조각 (tok/parent_local/
              sib_order/valid — 텐서 행).
        p_dists: dict node_ctx -> target 분포 (dict tok->prob).
                 node_ctx = -1 (root 직후) 또는 뷰 로컬 노드 인덱스
                 (그 노드 수락 후 컨텍스트).
        q_dists: 동일 키 -> draft 제안 분포 (그 컨텍스트의 q_eff).
        root_p: p_dists[-1] (root 직후 target 분포).
        rng:    random.Random (재현성).

    Returns: (accepted_path 로컬 인덱스 리스트, 종료토큰) — 종료토큰은
             recovery/bonus로 뽑힌 실제 커밋 토큰.
    """
    n = int(view["valid"])
    kids = {}
    for j in range(n):
        par = int(view["parent_local"][j])
        kids.setdefault(par, []).append(j)
    for v_ in kids.values():
        v_.sort(key=lambda j: int(view["sib_order"][j]))

    def norm(d):
        Z = sum(d.values())
        return {k: v / Z for k, v in d.items()} if Z > 1e-12 else {}

    path = []
    ctx = -1
    p = dict(root_p)
    while True:
        group = kids.get(ctx, [])
        q = dict(q_dists[ctx]) if group else None
        R = dict(p)
        accepted = None
        for j in group:
            tok = int(view["tok"][j])
            D = q
            if D.get(tok, 0.0) <= 0.0:
                raise ValueError("D_j[x_j]=0 — proposal/verifier parity 오류")
            a = min(1.0, R.get(tok, 0.0) / D[tok])
            if rng.random() < a:
                accepted = j
                break
            resid = {k: max(0.0, R.get(k, 0.0) - D.get(k, 0.0))
                     for k in set(R) | set(D)}
            R = norm(resid)
            if not R:
                R = {}                      # 도달불가 분기 (질량 0)
            q = norm({k: v for k, v in D.items() if k != tok})
        if accepted is None:
            # 전원 기각 (또는 자식 없음=잎): R(잔차 또는 plain p)에서 종료토큰
            src = R if group else p
            if not src:
                src = p                     # 잔차 소진 극단 — plain p
            r = rng.random()
            acc = 0.0
            last = None
            for k in sorted(src):
                acc += src[k]
                last = k
                if r <= acc:
                    return path, k
            return path, last
        path.append(accepted)
        ctx = accepted
        p = dict(p_dists[ctx])              # 컨텍스트 리셋 (리뷰4)


# --- T2.3: 트리 응답 wire 블록 (global max-padded — v6/리뷰4) ---

def tree_wire_ints_len(nv: int) -> int:
    """seq당 int64 블록 길이: [valid | u_valid | 예약(epoch)] + 4×nv."""
    return 3 + 4 * nv


def pack_tree_ints(view, hit_root: int, nv: int) -> torch.Tensor:
    """hit root의 뷰를 고정 길이 int64 블록으로. hit_root<0 → 전부 0
    (miss/P1 step — max-padded 규약상 크기는 동일)."""
    out = torch.zeros(tree_wire_ints_len(nv), dtype=torch.int64)
    if hit_root < 0:
        return out
    r = hit_root
    out[0] = view["valid"][r]
    out[1] = view["u_valid"][r]
    out[2] = 1                                   # epoch/버전 자리 (T2.4)
    o = 3
    out[o:o + nv] = view["tok"][r]
    out[o + nv:o + 2 * nv] = view["parent_local"][r]
    out[o + 2 * nv:o + 3 * nv] = view["sib_order"][r]
    out[o + 3 * nv:o + 4 * nv] = view["parent_q_ref"][r]
    return out


def parse_tree_ints(buf: torch.Tensor, nv: int):
    """→ dict (valid/u_valid/epoch/tok/parent_local/sib_order/pq_ref)."""
    o = 3
    return {
        "valid": int(buf[0]), "u_valid": int(buf[1]), "epoch": int(buf[2]),
        "tok": buf[o:o + nv],
        "parent_local": buf[o + nv:o + 2 * nv],
        "sib_order": buf[o + 2 * nv:o + 3 * nv],
        "parent_q_ref": buf[o + 3 * nv:o + 4 * nv],
    }


# --- T3.2: target-측 verify 행 조립 (pure — 엔진 배선은 T3.4) ---

def build_verify_rows(tree_ints, nv: int, pos0: int, block_table,
                      block_size: int):
    """트리 응답 블록 → target verify 행 텐서들 (v6 §7.5/리뷰4 row 계약).

    row 0..valid-1 = 뷰 노드 (scratch 셀: pos0+1+j — 리뷰4 계약),
    rope = pos0 + 1 + depth(노드) (depth는 parent_local 사슬로 복원).
    반환 dict: input_ids[valid], rope[valid], slot[valid], depth[valid],
    kids(보행용 인접), ancestors(행별 조상 행 목록 — mask용).
    """
    valid = tree_ints["valid"]
    tok = tree_ints["tok"][:valid]
    par = tree_ints["parent_local"][:valid]
    depth = torch.zeros(valid, dtype=torch.int64)
    ancestors = [[] for _ in range(valid)]
    for j in range(valid):
        p = int(par[j])
        if p >= 0:
            depth[j] = depth[p] + 1
            ancestors[j] = ancestors[p] + [p]
    rope = pos0 + 1 + depth
    scratch_pos = pos0 + 1 + torch.arange(valid)
    blk = block_table[(scratch_pos // block_size).long()]
    slot = blk * block_size + (scratch_pos % block_size)
    return {"input_ids": tok, "rope": rope, "slot": slot.to(torch.int64),
            "depth": depth, "ancestors": ancestors,
            "parent_local": par}


def build_verify_mask_packed(valid: int, ancestors, kv_len: int):
    """target tree-verify custom mask (FlashInfer packed, qo=valid 행).

    행 j 가시성: [프리픽스(0..pos0 포함 = kv_len−valid 이전 전부) |
    조상 행들의 scratch 셀 | 자기 셀]. kv_len = 프리픽스+valid.
    """
    prefix = kv_len - valid
    m = np.zeros((valid, kv_len), dtype=np.uint8)
    m[:, :prefix] = 1
    for j in range(valid):
        for a in ancestors[j]:
            m[j, prefix + a] = 1
        m[j, prefix + j] = 1
    packed = np.packbits(m.ravel(), bitorder="little")
    return torch.from_numpy(packed)


def tree_verify_walk_tensor(tree_ints, p_logits, q_parent_probs, temp_p,
                            coin_fn, mult_fn):
    """프로덕션 보행 (T3.4-a — full-vocab 텐서; 참조 tree_verify_walk와
    동일-코인 동등성 테스트로 고정).

    Args:
        tree_ints: parse_tree_ints 출력 (valid/tok/parent_local/
                   sib_order/parent_q_ref).
        p_logits:  [valid+1, V] — row0 = root 직후 컨텍스트의 target
                   logits, row j+1 = 노드 j 수락 후 컨텍스트.
        q_parent_probs: [U, V] — 부모 분포 (이미 temp/sampler_x 처리된
                   **확률**; draft 샘플측과 동일 build 함수 산출).
        temp_p:    target temperature (>0 — temp0은 트리 게이트).
        coin_fn(): U(0,1) 1개 (재현성 주입).
        mult_fn(probs): 분포에서 토큰 1개 샘플 (주입).

    Returns: (accepted_path 노드 인덱스 리스트, 종료 토큰 int)
    """
    valid = tree_ints["valid"]
    par = tree_ints["parent_local"]
    sib = tree_ints["sib_order"]
    kids = {}
    for j in range(valid):
        kids.setdefault(int(par[j]), []).append(j)
    for v_ in kids.values():
        v_.sort(key=lambda j: int(sib[j]))

    def p_of(ctx):
        row = 0 if ctx < 0 else ctx + 1
        return torch.softmax(p_logits[row].float() /
                             max(temp_p, 1e-8), dim=-1)

    path = []
    ctx = -1
    while True:
        group = kids.get(ctx, [])
        p = p_of(ctx)
        if not group:                          # 수락된 잎 → plain p bonus
            return path, int(mult_fn(p))
        R = p.clone()
        D = q_parent_probs[int(tree_ints["parent_q_ref"][group[0]])] \
            .float().clone()
        accepted = None
        for j in group:
            t = int(tree_ints["tok"][j])
            if float(D[t]) <= 0.0:
                raise ValueError("D_j[x_j]=0 — parity 오류 (리뷰4 규약)")
            a = min(1.0, float(R[t]) / float(D[t]))
            if coin_fn() < a:
                accepted = j
                break
            R = torch.clamp(R - D, min=0.0)
            Z = float(R.sum())
            R = R / Z if Z > 1e-12 else torch.zeros_like(R)
            D[t] = 0.0
            Zd = float(D.sum())
            D = D / Zd if Zd > 1e-12 else torch.zeros_like(D)
        if accepted is None:                   # 전원 기각 → 잔차 recovery
            src = R if float(R.sum()) > 1e-12 else p
            return path, int(mult_fn(src))
        path.append(accepted)
        ctx = accepted


def commit_copy_plan(accepted_path, pos0: int, block_table,
                     block_size: int):
    """T3.5-a — 수락 경로 KV의 scratch→canonical 복사 계획 (pure).

    scratch 셀 j는 pos0+1+j (build_verify_rows 계약), canonical 목적지는
    pos0+1+k (k = 경로 순서). 겹침 대비: (src, dst) 쌍 리스트 반환 —
    실행부(T3.5-b)는 rank별 temp buffer 경유 gather→scatter (리뷰4).
    """
    plan = []
    for k, j in enumerate(accepted_path):
        src_pos = pos0 + 1 + int(j)
        dst_pos = pos0 + 1 + k
        if src_pos == dst_pos:
            continue
        sb = int(block_table[src_pos // block_size]) * block_size \
            + src_pos % block_size
        db = int(block_table[dst_pos // block_size]) * block_size \
            + dst_pos % block_size
        plan.append((sb, db))
    return plan
