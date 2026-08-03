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
                      c_tensor, nv, beta, depth_cap, sample_fn):
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
    for f in range(F_total):
        sel = select_nodes(pool, policy, W, f, depth_cap)
        if not sel:
            eval_log.append(([], None))
            continue
        pri = pool.logpri[torch.tensor(sel)]
        roots = pool.root[torch.tensor(sel)]
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
                pool.add(int(toks[k][c]), i, cell,
                         int(pool.depth[i]) + 1, int(pool.root[i]), c,
                         float(pool.logpri[i])
                         + float(torch.log(torch.clamp(raws[k][c], min=1e-9))),
                         float(raws[k][c]))
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
                F_x=None, pad_token=0):
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
    for f in range(F_total):
        sel = select_nodes(pool, policy, W, f, depth_cap)
        n_sel = len(sel)
        fan = torch.zeros(W, dtype=torch.int64)
        if n_sel:
            pri = pool.logpri[torch.tensor(sel)]
            roots = pool.root[torch.tensor(sel)]
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
            cell_logits = torch.zeros(F_total * W, logits.shape[-1],
                                      dtype=logits.dtype)
        cell_logits[f * W:(f + 1) * W] = logits[:W]
        toks, raws = tree_sample_wor(logits, temps, c_tensor,
                                     sampler_x=sampler_x, F=F_x)
        for k, i in enumerate(sel):
            cell = f * W + k
            pool.cell[i] = cell
            pool.state[i] = 1
            for c in range(int(fan[k])):
                pool.add(int(toks[k][c]), i, cell,
                         int(pool.depth[i]) + 1, int(pool.root[i]), c,
                         float(pool.logpri[i])
                         + float(torch.log(torch.clamp(raws[k][c],
                                                       min=1e-9))),
                         float(raws[k][c]))
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
        pq_logits = torch.zeros(R, nv, V, dtype=cell_logits.dtype)
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
