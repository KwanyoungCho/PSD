"""P2-tree helpers — T1 (docs/duet/internal/20; 설계 docs/duet/internal/15 v6).

T1.2: 사전 예산 배분 (결정 ⑤v2 + 외부 리뷰 4차 규약).
무손실 핵심 규약 (D10): 위상/예산 결정은 샘플 정체를 관측하기 **전에**
확정한다 — 이 모듈의 함수들은 자식 토큰 정체를 입력으로 받지 않는다
(시그니처 수준에서 D10 보장).

전부 pure 함수 (텐서 in → 텐서 out, 상태 없음) — CPU 유닛테스트 대상.
"""
import numpy as np
import math
import os
import torch

from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling


def filter_unservable_tree_matches(match: torch.Tensor,
                                   cache_is_tree: torch.Tensor | None,
                                   batch_size: int) -> torch.Tensor:
    """Remove B=1-only tree cache rows from a wider-batch lookup.

    Dynamic-tree production rows deliberately omit chain-projection logits.
    They therefore must never be served through the ordinary B>1 payload.
    The returned tensor may alias ``match`` for B=1/no-metadata cases and is
    otherwise a new boolean tensor, keeping the cache lookup source intact.
    """
    if int(batch_size) == 1 or cache_is_tree is None:
        return match
    if match.ndim != 2 or cache_is_tree.ndim != 1 \
            or match.shape[1] != cache_is_tree.numel():
        raise ValueError(
            "tree cache match metadata shape mismatch: "
            f"match={tuple(match.shape)} tree={tuple(cache_is_tree.shape)}")
    return match & ~cache_is_tree.to(
        device=match.device, dtype=torch.bool).unsqueeze(0)


def tree_response_logit_rows(tree_valid: int, phase_source: int,
                             chain_rows: int, p1_cap: int,
                             p2_cap: int) -> tuple[int, int]:
    """Return ``(ordinary_q_rows, parent_q_rows)`` for one B=1 response.

    The two logit payloads are mutually exclusive.  A chain/miss response
    needs the ordinary K-wide q tensor, while a dynamic-tree response uses
    only its parent-q sidecar.  Sending both unconditionally made tree-enabled
    DUET transfer ``K + max(N1,N2)`` full-vocabulary rows on every request,
    including misses.  The fused metadata is received first, so both peers
    can deterministically choose the same following NCCL payload.  The
    persistent parent-q receive buffer remains phase-cap sized, but only the
    exact ``tree_valid`` prefix crosses the wire.
    """
    tree_valid = int(tree_valid)
    phase_source = int(phase_source)
    chain_rows = int(chain_rows)
    if tree_valid <= 0:
        return chain_rows, 0
    if phase_source == 1:
        cap = int(p1_cap)
    elif phase_source == 2:
        cap = int(p2_cap)
    else:
        raise ValueError(
            "active dynamic tree requires phase_source 1 or 2; "
            f"got phase={phase_source}, valid={tree_valid}")
    if tree_valid > cap:
        raise ValueError(
            f"tree valid={tree_valid} exceeds phase {phase_source} cap={cap}")
    return 0, tree_valid


def pack_tree_verify_mask(mask: torch.Tensor) -> torch.Tensor:
    """Pack a B=1 target tree mask in FlashInfer's little-endian layout.

    With one sequence, segmented packbits has one segment and is exactly a
    flattened 2-D pack.  Keeping this part on CPU also avoids launching a
    small GPU packing kernel inside every target-side ``plan()`` call.
    """
    if mask.device.type != "cpu" or mask.ndim != 2:
        raise ValueError("tree verify mask must be a 2-D CPU tensor")
    return torch.from_numpy(np.packbits(
        mask.numpy().reshape(-1), bitorder="little"))


def pack_tree_verify_mask_direct(ancestors, valid: int, rows: int,
                                 prefix_len: int, kv_len: int) \
        -> torch.Tensor:
    """Build the B=1 target tree mask directly in packed form on CPU.

    Materialising ``[rows, kv_len]`` booleans and then calling ``packbits``
    dominated the small target tree setup.  The mask is sparse and has a
    simple contract: every row sees the common prefix, while each real tree
    node additionally sees its ancestors and itself.  Set those ranges/bits
    directly in FlashInfer's little-endian packed layout instead.
    """
    valid = int(valid)
    rows = int(rows)
    prefix_len = int(prefix_len)
    kv_len = int(kv_len)
    if not (0 <= valid < rows):
        raise ValueError(f"valid={valid} must be in [0, rows={rows})")
    if not (0 <= prefix_len <= kv_len):
        raise ValueError(
            f"prefix_len={prefix_len} must be in [0, kv_len={kv_len}]")
    if len(ancestors) < valid:
        raise ValueError("ancestor list is shorter than valid node count")

    nbits = rows * kv_len
    packed = np.zeros((nbits + 7) // 8, dtype=np.uint8)

    def _set_range(start, end):
        """Set the half-open flattened bit range [start,end)."""
        if start >= end:
            return
        first, first_bit = divmod(start, 8)
        last, last_bit = divmod(end, 8)
        if first == last:
            packed[first] |= np.uint8(
                ((1 << (last_bit - first_bit)) - 1) << first_bit)
            return
        if first_bit:
            packed[first] |= np.uint8((0xFF << first_bit) & 0xFF)
            first += 1
        if first < last:
            packed[first:last] = 0xFF
        if last_bit:
            packed[last] |= np.uint8((1 << last_bit) - 1)

    for row in range(rows):
        base = row * kv_len
        _set_range(base, base + prefix_len)
    for node in range(valid):
        base = (node + 1) * kv_len
        for ancestor in ancestors[node]:
            col = prefix_len + int(ancestor)
            packed[(base + col) >> 3] |= np.uint8(1 << ((base + col) & 7))
        col = prefix_len + node
        packed[(base + col) >> 3] |= np.uint8(1 << ((base + col) & 7))
    return torch.from_numpy(packed)


def allocate_tree_p1_fanouts(parent_local, sib_order, raw_q,
                              total_budget: int,
                              chain_fanouts) -> list[int]:
    """Allocate a tree-hit P1 budget without weakening the chain backbone.

    A P2 tree has more possible terminal contexts than the K2 chain.  The old
    code redistributed the whole P1 budget across those contexts, so a
    backbone context that received two candidates in the chain could receive
    only one after a tree hit.  That made P1 quality depend on whether the
    previous P2 response happened to be a tree.

    This allocator first gives the root/backbone contexts exactly the same
    per-position minimum as the short-chain P1 layout.  Every remaining tree
    context then gets one lane while capacity permits.  Leftover lanes are
    assigned by cumulative draft confidence with a diminishing-return divisor.
    No target result or host-side tuning parameter is needed, so P1 can still
    execute while the target is producing the proxy message.

    Context numbering is the cache-key convention used by ``DraftRunner``:
    context 0 is the recovery/root context and context ``1+j`` is tree node
    ``j``.  The backbone is the first-child (``sib_order == 0``) chain.
    """
    par = [int(x) for x in parent_local]
    sib = [int(x) for x in sib_order]
    q = [float(x) for x in raw_q]
    n = len(par)
    if len(sib) != n or len(q) != n:
        raise ValueError("tree P1 topology fields must have equal length")
    if total_budget <= 0:
        raise ValueError("tree P1 total_budget must be positive")

    n_ctx = n + 1
    counts = [0] * n_ctx

    # Build the first-child chain in actual node-id space.  Stable generation
    # order makes the first matching sib=0 child the canonical backbone.
    first_child = {}
    for j, (p, s) in enumerate(zip(par, sib)):
        if s == 0 and p not in first_child:
            first_child[p] = j
    backbone_ctx = [0]
    parent = -1
    for _ in range(max(0, len(chain_fanouts) - 1)):
        child = first_child.get(parent)
        if child is None:
            break
        backbone_ctx.append(1 + child)
        parent = child

    required = 0
    for depth, ctx in enumerate(backbone_ctx):
        want = max(0, int(chain_fanouts[depth]))
        if required + want > total_budget:
            want = total_budget - required
        counts[ctx] = want
        required += want
        if required == total_budget:
            return counts

    # Cumulative EAGLE2-style draft confidence for each reachable context.
    conf = [1.0] * n_ctx
    for j, p in enumerate(par):
        parent_conf = 1.0 if p < 0 else conf[1 + p]
        qj = q[j]
        if not math.isfinite(qj) or qj < 0.0:
            qj = 0.0
        conf[1 + j] = parent_conf * min(qj, 1.0)

    left = total_budget - required
    backbone_set = set(backbone_ctx)
    # Preserve cache coverage for sibling terminal contexts before adding a
    # third/fourth candidate to an already covered context.
    for ctx in sorted((c for c in range(n_ctx)
                       if c not in backbone_set),
                      key=lambda c: (-conf[c], c)):
        if left == 0:
            break
        counts[ctx] = 1
        left -= 1

    # Parameter-free diminishing returns.  This is equivalent to repeatedly
    # taking the largest next confidence contribution rather than applying a
    # fixed depth/terminal-probability heuristic.
    while left > 0:
        ctx = max(range(n_ctx),
                  key=lambda c: (conf[c] / (counts[c] + 1), -c))
        counts[ctx] += 1
        left -= 1
    return counts


def sanitize_root_inputs(root_toks: torch.Tensor,
                         root_piv: torch.Tensor,
                         rope_base: torch.Tensor,
                         vocab_size: int,
                         max_position: int):
    """Return fixed-shape, model-safe root inputs without host readback.

    Invalid roots remain as padding lanes, but token/position become zero and
    probability becomes zero.  The zero probability is the authoritative
    allocation mask used by both the arena and the captured executor.
    """
    valid = ((root_toks >= 0) & (root_toks < vocab_size)
             & (rope_base >= 0) & (rope_base < max_position)
             & torch.isfinite(root_piv) & (root_piv > 0))
    return (
        torch.where(valid, root_toks, torch.zeros_like(root_toks)),
        torch.where(valid, root_piv, torch.zeros_like(root_piv)),
        torch.where(valid, rope_base, torch.zeros_like(rope_base)),
        valid,
    )


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
    # 이슈 #24: piv<=0은 예산 제외 sentinel (root_count 분리 — cap
    # 포화 후 water-filling이 무자격 root로 새지 않게 명시 배제)
    _elig = piv > 0
    w = piv.clamp_min(1e-9).double().pow(beta)
    w = torch.where(_elig, w, torch.zeros_like(w))
    if not bool(_elig.any()):
        return torch.zeros(R, dtype=torch.int64)
    # 이슈 #33 (리뷰2-7): cap-절단 질량을 active set에 '비례'로 재배분
    # (water-filling 본형). 종전 frac 라운드-로빈은 잔여를 균등화해
    # [0.9,.09,.01]·β=1·total16·cap8 → [8,5,3] (비례 기대 [8,7,1]) —
    # 약root 과잉지원이 P_iv<0.01 슬롯 점유(41.5%, hit 기여 3.9%)의
    # 한 원인. 포화 root를 제외하며 남은 예산을 남은 가중치 비율로
    # 재계산: 수렴 ≤ R회. 소진 보장(sum == min(total, R_elig·cap))은
    # #23과 동일 (정수 반올림은 largest-remainder + 라운드-로빈 백스톱).
    quota = torch.zeros_like(w)
    active = _elig.clone()
    left = float(total)
    for _ in range(R):
        if left <= 1e-9 or not bool(active.any()):
            break
        wa = torch.where(active, w, torch.zeros_like(w))
        add = wa / wa.sum() * left
        newq = quota + add
        over = active & (newq >= cap)
        if bool(over.any()):
            left -= float((cap - quota[over]).sum())
            quota[over] = float(cap)
            active[over] = False
        else:
            quota = newq
            left = 0.0
    base = quota.floor().long().clamp_max(cap)
    rem_budget = int(total - base.sum().item())
    n_elig_cap = int(_elig.sum().item()) * cap
    rem_budget = min(rem_budget, n_elig_cap - int(base.sum().item()))
    if rem_budget > 0:
        frac = quota - quota.floor()
        order = torch.argsort(frac, descending=True, stable=True).tolist()
        progressed = True
        while rem_budget > 0 and progressed:
            progressed = False
            for i in order:
                if rem_budget <= 0:
                    break
                if base[i] < cap and bool(_elig[i]):
                    base[i] += 1
                    rem_budget -= 1
                    progressed = True
    return base


def alloc_policy_root_budgets(piv: torch.Tensor, policy: str, total: int,
                              beta: float, cap: int) -> torch.Tensor:
    """Allocate root-local *stored-node* budgets for a rollout policy.

    ``total`` is the number of parent forward cells (``F * W``).  It is not
    generally a bound on the number of children that can be retained: one
    evaluated parent produces up to ``C`` ordered WOR children in the same
    model forward.  The historical confidence policy conflated those two
    quantities and paid for branching by deleting low-ranked roots.

    The production ``dynamic`` policy and its legacy ``eagle`` spelling give
    every live root the same response-capacity bound, evaluate all roots in
    round zero, and globally choose later parents by cumulative confidence.
    ``coverage``/``backbone`` retain their historical fixed-depth behavior
    only for controlled reproduction.  Zero-prior padding roots stay at
    budget zero.

    The draw identities are deliberately not inputs.  Budget/topology is
    fixed before WOR sampling, preserving the lossless verifier contract.
    """
    if policy in (
            "coverage", "backbone", "dynamic", "eagle", "hybrid",
            "adaptive"):
        return torch.where(
            piv > 0,
            torch.full_like(piv, int(cap), dtype=torch.int64),
            torch.zeros_like(piv, dtype=torch.int64),
        )
    budget_beta = 0.5 if policy == "confidence" else beta
    return alloc_root_budgets(piv, total=total, beta=budget_beta, cap=cap)


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


def alloc_fanouts_global(parent_priority: torch.Tensor,
                         parent_root: torch.Tensor,
                         root_remaining: torch.Tensor,
                         c_tensor: int,
                         future_rounds: int) -> torch.Tensor:
    """Fanout for EAGLE-style global expansion with a final-view cap.

    Every selected parent receives one child before a parent receives a
    second/third child.  For each root, nodes needed to keep one selected path
    extendable through the remaining rounds are reserved.  Unlike the old
    backbone policy this reserve is created only for roots that actually win
    the global frontier, never for every initial root.
    """
    W = parent_priority.numel()
    out = torch.zeros(W, dtype=torch.int64)
    order = torch.argsort(parent_priority, descending=True, stable=True)
    rem = root_remaining.clone()
    future = max(0, int(future_rounds))
    now = torch.where(
        rem > 0, torch.maximum(torch.ones_like(rem), rem - future),
        torch.zeros_like(rem))
    # Breadth first: every useful selected parent gets one child.
    for i in order.tolist():
        r = int(parent_root[i])
        if int(now[r]) > 0:
            out[i] = 1
            now[r] -= 1
    # Then spend remaining current-round capacity by confidence.
    for _ in range(max(0, c_tensor - 1)):
        for i in order.tolist():
            r = int(parent_root[i])
            if int(now[r]) > 0 and int(out[i]) < c_tensor:
                out[i] += 1
                now[r] -= 1
    return out


def alloc_fanouts_backbone(parent_priority: torch.Tensor,
                           parent_root: torch.Tensor,
                           root_remaining: torch.Tensor,
                           root_reserve: torch.Tensor,
                           is_tip: torch.Tensor,
                           c_tensor: int) -> torch.Tensor:
    """backbone-우선 fanout (형상 진단 2026-08-04 — docs/duet/internal/21 §4.5).

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


def alloc_fanouts_adaptive(parent_priority: torch.Tensor,
                           parent_root: torch.Tensor,
                           root_remaining: torch.Tensor,
                           root_reserve: torch.Tensor,
                           is_tip: torch.Tensor,
                           c_tensor: int) -> torch.Tensor:
    """Keep every first-child chain; add siblings above mean path mass.

    The score is ``log P_proxy(root) + sum(log q(previous child))`` and is
    known before the current token draw.  Mandatory tips therefore remain
    chain-equivalent, while optional width is parameter-free and cannot bias
    the ordered residual samples by inspecting their identities.
    """
    out = alloc_fanouts_backbone(
        parent_priority, parent_root, root_remaining, root_reserve,
        is_tip, c_tensor)
    live = is_tip & torch.isfinite(parent_priority)
    n_live = int(live.sum())
    if n_live == 0:
        return torch.zeros_like(out)
    threshold = torch.logsumexp(parent_priority[live], 0) \
        - torch.log(torch.tensor(float(n_live),
                                 dtype=parent_priority.dtype))
    extra_ok = torch.isfinite(parent_priority) \
        & (parent_priority >= threshold)
    # A weak tip keeps exactly its mandatory child.  Weak optional rescue
    # parents produce no children, leaving the root-local view smaller.
    floor = is_tip.long()
    return torch.where(extra_ok, out, torch.minimum(out, floor))


def terminal_mass_dp(par, alpha):
    """종단질량 DP — **조건부 α 입력 규약** (이슈 #28 정정).

    reach(j) = reach(parent)·∏앞형제(1−α_j)·α_j;
    terminal(ctx) = reach(ctx)·∏자식(1−α). 곱 형태가 정확하려면 α_k가
    "앞 형제 전원 기각 given" **조건부** 수락확률이어야 한다 — 원본
    p/q로 계산한 독립 α를 넣으면 2형제 반례에서 all-reject 질량 0
    (실제 0.5) — 리뷰2-2 확정. 라이브 경로는 tree_policy_b_ladder가
    조건부 α·잔차까지 일괄 계산하므로 이 함수를 더 쓰지 않는다;
    분석 도구(e1_explicit_tree — λ-할인 조건부 α 모델)와 체인-퇴화
    검증용으로 유지.

    Args:
        par:   [valid] parent_local (-1=rec 직결; 생성 순서 = 부모 선행).
        alpha: [valid] 노드별 **조건부** α.
    Returns: term [valid+1] — [0]=rec ctx, [1+j]=노드 j에서 종단.
    """
    valid = len(par)
    kids = {}
    for j in range(valid):
        kids.setdefault(int(par[j]), []).append(j)
    reach = [0.0] * valid
    term = [0.0] * (valid + 1)

    def _terminal_of(ctx_key, rv):
        m = rv
        for c in kids.get(ctx_key, []):
            m *= (1.0 - float(alpha[c]))
        return m

    term[0] = _terminal_of(-1, 1.0)
    for j in range(valid):
        pk = int(par[j])
        base = 1.0 if pk < 0 else reach[pk]
        pre = 1.0
        for sblg in kids.get(pk, []):
            if sblg == j:
                break
            pre *= (1.0 - float(alpha[sblg]))
        reach[j] = base * pre * float(alpha[j])
        term[1 + j] = _terminal_of(j, reach[j])
    return torch.tensor(term, dtype=torch.float32)


def tree_policy_b_ladder(par, sib, tokens, p_rows, q_rows):
    """이슈 #28+#34: 트리 Policy-B의 **정확 sibling 사다리** (전면 텐서).

    tree_verify_walk_tensor의 기각 갱신을 그대로 미러한다:
      a_k = min(1, R_k[t_k]/D_k[t_k]) (조건부 — 앞 형제 전원 기각 given),
      기각 시 R ← norm((R−D)+), D[t_k] ← 0 후 renorm,
      전원 기각 ctx의 recovery 원천 = 최종 R (합 소멸 시 p^E 폴백).
    종전 terminal_mass_dp 독립-α 근사(원본 p/q로 모든 형제 평가)는
    2형제 반례에서 all-reject 질량 0 vs 실제 0.28125 — 사다리로 대체.

    전 연산 텐서 op (`.item()`/`.cpu()` 없음) — GPU 입력이면 sync 0회
    (#34: 종전 파이썬 구현이 exit_logits 0.3→17.6ms). CPU 입력이면
    유닛테스트 참조 구현으로 동작.

    Args:
        par:    [valid] python list — parent_local (-1=rec 직결).
        sib:    [valid] python list — 형제 순서 (walk와 동일 정렬 키).
        tokens: [valid] int64 텐서 — 노드 토큰 (p/q와 같은 디바이스).
        p_rows: [valid+1, V] — p^E 확률 (row 0=rec ctx, 1+j=노드 j).
        q_rows: [valid, V] — 노드별 **부모 컨텍스트의** draft 제안 q.

    Returns:
        alpha  [valid]    — 조건부 수락확률 (terminal DP 입력 규약).
        term   [valid+1]  — 컨텍스트별 종단질량 (합 1).
        resid  [valid+1, V] — ctx별 recovery 분포 (정규화; 자식 있는
            ctx는 사다리 최종 R, 잎 ctx는 p^E row).
    """
    valid = len(par)
    dev = p_rows.device
    resid = p_rows[:valid + 1].clone()
    if valid == 0:
        return (torch.zeros(0, device=dev, dtype=p_rows.dtype),
                torch.ones(1, device=dev, dtype=p_rows.dtype),
                resid)
    kids = {}
    for j in range(valid):
        kids.setdefault(int(par[j]), []).append(j)
    for v_ in kids.values():
        v_.sort(key=lambda j: int(sib[j]))
    groups = sorted(kids.keys())                     # ctx key 오름차순
    G = len(groups)
    max_c = max(len(kids[g]) for g in groups)
    depth = [0] * valid
    for j in range(valid):
        depth[j] = 1 if int(par[j]) < 0 else depth[int(par[j])] + 1
    # 단일 H2D 토포 팩 (#34 μ-opt): 분산 torch.tensor(..., device) 호출
    # (~8회)은 혼잡 스트림에서 staging 직렬화 — CPU에서 한 텐서로 조립
    # 후 1회 전송, GPU에서는 view 슬라이스만.
    # 레이아웃: [ctx_rows G | first_child G | cj G*C | par valid]
    # cj pad = valid (dummy slot — alpha/presib를 valid+1로 잡고 버림).
    cj_pad = []
    for g in groups:
        ch = kids[g]
        cj_pad += ch + [valid] * (max_c - len(ch))
    topo = torch.tensor(
        [g + 1 for g in groups]
        + [kids[g][0] for g in groups]
        + cj_pad
        + [int(x) for x in par],
        dtype=torch.int64).to(dev, non_blocking=True)
    ctx_rows = topo[:G]
    first_child = topo[G:2 * G]
    cj_mat = topo[2 * G:2 * G + G * max_c].view(G, max_c)
    par_t = topo[2 * G + G * max_c:]
    valid_mat = cj_mat < valid                       # [G, C] 유효 마스크
    tok_ext = torch.cat([tokens, tokens.new_zeros(1)])
    alpha_x = torch.zeros(valid + 1, device=dev,
                          dtype=p_rows.dtype)         # [+dummy]
    presib_x = torch.ones(valid + 1, device=dev, dtype=p_rows.dtype)
    R_mat = p_rows.index_select(0, ctx_rows).clone()   # [G, V]
    D_mat = q_rows.index_select(0, first_child).clone()
    group_cum = torch.ones(G, 1, device=dev, dtype=p_rows.dtype)
    for s in range(max_c):
        cj = cj_mat[:, s]                            # [G] (pad=valid)
        vm = valid_mat[:, s:s + 1].float()           # [G,1]
        tj = tok_ext.index_select(0, cj).unsqueeze(1)   # [G,1]
        r_t = R_mat.gather(1, tj)
        d_t = D_mat.gather(1, tj)
        a_s = (r_t / (d_t + 1e-10)).clamp(max=1.0) * vm
        alpha_x.index_copy_(0, cj, a_s.squeeze(1))
        presib_x.index_copy_(0, cj, group_cum.squeeze(1))
        group_cum = group_cum * (1.0 - a_s)
        # 기각 갱신 (walk 동일: R←norm((R−D)+); D[t]←0 renorm) — 전 G
        # 행 dense 계산 후 무효 행은 where로 원복 (subset index 제거)
        R_new = (R_mat - D_mat).clamp(min=0.0)
        Z = R_new.sum(-1, keepdim=True)
        R_new = torch.where(Z > 1e-12, R_new / Z.clamp_min(1e-30),
                            torch.zeros_like(R_new))
        R_mat = torch.where(vm > 0, R_new, R_mat)
        D_new = D_mat.scatter(1, tj, 0.0)
        Zd = D_new.sum(-1, keepdim=True)
        D_new = torch.where(Zd > 1e-12, D_new / Zd.clamp_min(1e-30),
                            torch.zeros_like(D_new))
        D_mat = torch.where(vm > 0, D_new, D_mat)
    alpha = alpha_x[:valid]
    presib = presib_x[:valid]
    # recovery 원천 (walk: src = R if sum>1e-12 else p)
    R_ok = R_mat.sum(-1, keepdim=True) > 1e-12
    resid[ctx_rows] = torch.where(
        R_ok, R_mat, p_rows.index_select(0, ctx_rows))
    # reach: reach[j] = alpha·presib·reach[parent] — depth 고정점 반복
    base = alpha * presib
    reach = base.clone()
    one = torch.ones(1, device=dev, dtype=p_rows.dtype)
    for _ in range(max(depth) - 1):
        reach = base * torch.cat([one, reach]).index_select(0, par_t + 1)
    reach_ext = torch.cat([one, reach])              # [valid+1]
    allrej = torch.ones(valid + 1, device=dev, dtype=p_rows.dtype)
    allrej.index_copy_(0, ctx_rows, group_cum.squeeze(1))  # g=-1 → row 0
    term = reach_ext * allrej
    return alpha, term, resid


def pack_tree_proxy_topology(par, sib, nv: int, c_max: int = 3,
                             device=None):
    """Pack a dynamic tree into fixed-size tensors for graph replay.

    The values remain dynamic, but the shapes are always ``[nv+1,c_max]``
    and ``[nv]``.  This is the same fixed-envelope technique used by the P2
    draft executor: CUDA graph capture constrains addresses/shapes, not the
    parent/sibling values stored at those addresses.
    """
    par_l = [int(x) for x in par]
    sib_l = [int(x) for x in sib]
    valid = len(par_l)
    if valid > nv or len(sib_l) != valid:
        raise ValueError(
            f"tree proxy topology shape: valid={valid} nv={nv} "
            f"sib={len(sib_l)}")
    child = torch.full((nv + 1, c_max), nv, dtype=torch.int64)
    child_valid = torch.zeros((nv + 1, c_max), dtype=torch.bool)
    par_pad = torch.full((nv,), -1, dtype=torch.int64)
    sib_pad = torch.zeros((nv,), dtype=torch.int64)
    node_valid = torch.zeros((nv,), dtype=torch.bool)
    for j, (p, s) in enumerate(zip(par_l, sib_l)):
        if p < -1 or p >= j:
            raise ValueError(
                f"tree proxy parent invariant at node {j}: parent={p}")
        if s < 0 or s >= c_max:
            raise ValueError(
                f"tree proxy sibling capacity at node {j}: sib={s} "
                f"c_max={c_max}")
        row = p + 1
        if bool(child_valid[row, s]):
            raise ValueError(
                f"tree proxy duplicate sibling slot: parent={p} sib={s}")
        child[row, s] = j
        child_valid[row, s] = True
        par_pad[j] = p
        sib_pad[j] = s
        node_valid[j] = True
    out = {
        "child": child,
        "child_valid": child_valid,
        "par": par_pad,
        "sib": sib_pad,
        "node_valid": node_valid,
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def tree_policy_b_ladder_fixed(tokens, p_rows, q_rows, child,
                               child_valid, par, sib, node_valid,
                               depth_steps: int):
    """Fixed-shape equivalent of :func:`tree_policy_b_ladder`.

    ``p_rows`` is ``[Nv+1,V]`` and ``q_rows``/``tokens`` are padded to Nv.
    Invalid rows carry zero terminal mass.  All loops have config-fixed
    bounds, so this function can be captured once and replayed for arbitrary
    dynamic tree values without host work between the exit lm-head and P2
    proxy send.
    """
    n = tokens.shape[0]
    r = p_rows.shape[0]
    if r != n + 1 or q_rows.shape[0] != n:
        raise ValueError(
            f"fixed tree proxy rows: p={p_rows.shape} q={q_rows.shape} "
            f"tokens={tokens.shape}")
    c_max = child.shape[1]
    dtype, dev = p_rows.dtype, p_rows.device
    tok_ext = torch.cat([tokens, tokens.new_zeros(1)])
    q_ext = torch.cat([q_rows, torch.zeros_like(q_rows[:1])], dim=0)
    first = child[:, 0].clamp(min=0, max=n)
    R_mat = p_rows.clone()
    D_mat = q_ext.index_select(0, first).clone()
    has_child = child_valid[:, :1]
    D_mat = torch.where(has_child, D_mat, torch.zeros_like(D_mat))
    group_cum = torch.ones(r, 1, dtype=dtype, device=dev)
    alpha_cols, presib_cols = [], []
    for s in range(c_max):
        cj = child[:, s].clamp(min=0, max=n)
        vm = child_valid[:, s:s + 1].to(dtype)
        tj = tok_ext.index_select(0, cj).unsqueeze(1)
        r_t = R_mat.gather(1, tj)
        d_t = D_mat.gather(1, tj)
        a_s = (r_t / (d_t + 1e-10)).clamp(max=1.0) * vm
        alpha_cols.append(a_s.squeeze(1))
        presib_cols.append(group_cum.squeeze(1))
        group_cum = group_cum * (1.0 - a_s)
        R_new = (R_mat - D_mat).clamp(min=0.0)
        Z = R_new.sum(-1, keepdim=True)
        R_new = torch.where(Z > 1e-12, R_new / Z.clamp_min(1e-30),
                            torch.zeros_like(R_new))
        R_mat = torch.where(vm > 0, R_new, R_mat)
        D_new = D_mat.scatter(1, tj, 0.0)
        Zd = D_new.sum(-1, keepdim=True)
        D_new = torch.where(Zd > 1e-12, D_new / Zd.clamp_min(1e-30),
                            torch.zeros_like(D_new))
        D_mat = torch.where(vm > 0, D_new, D_mat)

    alpha_mat = torch.stack(alpha_cols, dim=1)
    presib_mat = torch.stack(presib_cols, dim=1)
    row = (par + 1).clamp(min=0, max=n)
    col = sib.clamp(min=0, max=c_max - 1)
    alpha = alpha_mat[row, col] * node_valid.to(dtype)
    presib = presib_mat[row, col]
    R_ok = R_mat.sum(-1, keepdim=True) > 1e-12
    resid = torch.where(R_ok, R_mat, p_rows)

    base = alpha * presib
    reach = base.clone()
    one = torch.ones(1, dtype=dtype, device=dev)
    parent_idx = (par + 1).clamp(min=0, max=n)
    for _ in range(max(0, int(depth_steps) - 1)):
        reach = base * torch.cat([one, reach]).index_select(0, parent_idx)
    reach = reach * node_valid.to(dtype)
    reach_ext = torch.cat([one, reach])
    context_valid = torch.cat([
        torch.ones(1, dtype=torch.bool, device=dev), node_valid])
    term = reach_ext * group_cum.squeeze(1) * context_valid.to(dtype)
    return alpha, term, resid


def tree_proxy_candidates_fixed(exit_logits, q_logits, tokens, topology,
                                wire_n: int, depth_steps: int,
                                top_k: int | None = None):
    """Fixed-shape, capture-safe tree proxy candidate computation.

    ``top_k`` preserves the established chain Policy-B score scale: each
    terminal context first keeps and renormalizes its top-k correction
    candidates, then contexts compete for the shared wire.  Ranking the full
    vocabulary directly is not chain-equivalent even for a one-child tree,
    because contexts with different top-k retained mass receive different
    relative scales.
    """
    p_rows = torch.softmax(exit_logits.float(), dim=-1)
    q_rows = torch.softmax(q_logits.float(), dim=-1)
    _alpha, term, resid = tree_policy_b_ladder_fixed(
        tokens, p_rows, q_rows,
        topology["child"], topology["child_valid"],
        topology["par"], topology["sib"], topology["node_valid"],
        depth_steps)
    V = resid.shape[-1]
    n = tokens.shape[0]
    exclude_idx = ((topology["par"] + 1).clamp(min=0, max=n) * V
                   + tokens.clamp(min=0, max=V - 1))
    if top_k is None:
        # Legacy/full-vocabulary reference used by analysis tests.  Keep its
        # exact arithmetic available; production always supplies top_k.
        piv_rows = resid * term.unsqueeze(1)
        flat_ext = torch.cat([
            piv_rows.reshape(-1), piv_rows.new_zeros(1)])
        exclude = torch.where(
            topology["node_valid"], exclude_idx,
            torch.full_like(exclude_idx, piv_rows.numel()))
        flat_ext.scatter_(0, exclude, 0.0)
        flat = flat_ext[:-1]
        k = min(int(wire_n), flat.numel())
        top_v, top_i = flat.topk(k)
        chosen_pos = (top_i // V).to(torch.int64)
        chosen_tok = pack_piv((top_i % V).to(torch.int64), top_v)
        return chosen_pos, chosen_tok, top_v

    flat_ext = torch.cat([resid.reshape(-1), resid.new_zeros(1)])
    exclude = torch.where(
        topology["node_valid"], exclude_idx,
        torch.full_like(exclude_idx, resid.numel()))
    flat_ext.scatter_(0, exclude, 0.0)
    correction = flat_ext[:-1].view(n + 1, V)
    ctx_k = min(int(top_k), V)
    correction_prob, correction_id = correction.topk(ctx_k, dim=-1)
    correction_prob = correction_prob / correction_prob.sum(
        -1, keepdim=True).clamp(min=1e-10)
    piv = correction_prob * term.unsqueeze(1)
    k = min(int(wire_n), piv.numel())
    top_v, top_i = piv.reshape(-1).topk(k)
    chosen_pos = (top_i // ctx_k).to(torch.int64)
    chosen_id = correction_id.reshape(-1).gather(0, top_i)
    chosen_tok = pack_piv(chosen_id.to(torch.int64), top_v)
    return chosen_pos, chosen_tok, top_v


class TreeProxyCUDAGraph:
    """One fixed-address CUDA graph for a target tree-proxy bucket."""

    @torch.inference_mode()
    def __init__(self, nv: int, vocab_size: int, wire_n: int,
                 depth_steps: int, dtype, device, top_k: int | None = None):
        self.nv = int(nv)
        self.V = int(vocab_size)
        self.wire_n = int(wire_n)
        self.depth_steps = int(depth_steps)
        self.top_k = (None if top_k is None else int(top_k))
        self.device = torch.device(device)
        self.in_exit = torch.zeros(
            self.nv + 1, self.V, dtype=dtype, device=self.device)
        self.in_q = torch.zeros(
            self.nv, self.V, dtype=dtype, device=self.device)
        self.in_tokens = torch.zeros(
            self.nv, dtype=torch.int64, device=self.device)
        topo = pack_tree_proxy_topology([], [], self.nv,
                                        device=self.device)
        self.topology = topo
        self.valid = 0

        # Warm every op before capture; allocation/compilation is forbidden
        # inside a first production replay.
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm):
            for _ in range(2):
                tree_proxy_candidates_fixed(
                    self.in_exit, self.in_q, self.in_tokens,
                    self.topology, self.wire_n, self.depth_steps,
                    self.top_k)
        warm.synchronize()
        # Capture failures leave CUDA's graph/RNG bookkeeping unusable for
        # ordinary eager execution in this process.  Synchronize explicitly
        # so any warm-up error is raised before capture starts.
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out_pos, self.out_tok, self.out_piv = \
                tree_proxy_candidates_fixed(
                    self.in_exit, self.in_q, self.in_tokens,
                    self.topology, self.wire_n, self.depth_steps,
                    self.top_k)

    @torch.inference_mode()
    def prepare_topology(self, par, sib):
        packed = pack_tree_proxy_topology(par, sib, self.nv)
        self.valid = len(par)
        for key, value in packed.items():
            self.topology[key].copy_(value.to(self.device))

    @torch.inference_mode()
    def replay(self, exit_logits, q_logits, tokens):
        """Copy dynamic values, replay, and return persistent wire buffers."""
        valid = int(self.valid)
        if exit_logits.shape[0] != valid + 1 \
                or q_logits.shape[0] < valid or tokens.shape[0] < valid:
            raise RuntimeError(
                "tree proxy graph input mismatch: "
                f"valid={valid} exit={tuple(exit_logits.shape)} "
                f"q={tuple(q_logits.shape)} tok={tuple(tokens.shape)}")
        self.in_exit.zero_()
        self.in_q.zero_()
        self.in_tokens.zero_()
        self.in_exit[:valid + 1].copy_(exit_logits[:valid + 1])
        if valid:
            self.in_q[:valid].copy_(q_logits[:valid])
            self.in_tokens[:valid].copy_(tokens[:valid])
        self.graph.replay()
        return self.out_pos, self.out_tok, self.out_piv


def chain_proxy_candidates_fixed(exit_logits, q_logits, tokens,
                                 top_k: int, wire_n: int,
                                 pack_scores: bool):
    """Capture-safe B=1 Policy-B proxy calculation for a chain.

    This is the fixed-shape equivalent of
    ``Verifier._compute_and_send_proxy``.  It deliberately preserves that
    function's row shapes and operation order: K early-exit rows are
    normalized separately from the final all-accepted row.  Keeping the
    exact chain policy here lets cache-miss, K1-hit and K2-hit steps use one
    graph replay instead of dozens of small PyTorch launches.
    """
    K, V = q_logits.shape
    p_e = torch.softmax(exit_logits[:K].float(), dim=-1)
    p_d = torch.softmax(q_logits.float(), dim=-1)
    gather = tokens[:K].view(K, 1)
    p_e_y = p_e.gather(1, gather).squeeze(1)
    p_d_y = p_d.gather(1, gather).squeeze(1)
    accept = (p_e_y / (p_d_y + 1e-10)).clamp(max=1.0)

    residual = (p_e - p_d).clamp(min=0)
    residual.scatter_(1, gather, 0.0)
    top_prob, top_id = residual.topk(int(top_k), dim=-1)
    top_prob = top_prob / top_prob.sum(-1, keepdim=True).clamp(min=1e-10)

    cumprod = torch.cumprod(accept, dim=0)
    h = torch.zeros(K + 1, dtype=accept.dtype, device=accept.device)
    h[0] = 1 - accept[0]
    if K > 1:
        h[1:K] = cumprod[:-1] * (1 - accept[1:])
    h[K] = cumprod[-1]

    p_last = torch.softmax(exit_logits[K].float(), dim=-1)
    last_prob, last_id = p_last.topk(int(top_k), dim=-1)
    last_prob = last_prob / last_prob.sum().clamp(min=1e-10)
    correction_prob = torch.cat([top_prob, last_prob.unsqueeze(0)], dim=0)
    correction_id = torch.cat([top_id, last_id.unsqueeze(0)], dim=0)
    piv = h.unsqueeze(1) * correction_prob
    top_v, top_i = piv.reshape(-1).topk(int(wire_n))
    chosen_pos = (top_i // int(top_k)).to(torch.int64)
    chosen_tok = correction_id.reshape(-1).gather(0, top_i)
    if pack_scores:
        chosen_tok = pack_piv(chosen_tok, top_v)
    return chosen_pos, chosen_tok, top_v


class ChainProxyCUDAGraph:
    """One fixed-address proxy graph for one chain length K."""

    @torch.inference_mode()
    def __init__(self, k: int, vocab_size: int, top_k: int, wire_n: int,
                 pack_scores: bool, dtype, device):
        self.k = int(k)
        self.V = int(vocab_size)
        self.top_k = int(top_k)
        self.wire_n = int(wire_n)
        self.pack_scores = bool(pack_scores)
        self.device = torch.device(device)
        if self.k <= 0:
            raise ValueError("chain proxy graph K must be positive")
        if self.wire_n > (self.k + 1) * self.top_k:
            raise ValueError(
                "chain proxy wire exceeds candidate count: "
                f"wire_n={self.wire_n}, candidates="
                f"{(self.k + 1) * self.top_k}")
        self.in_exit = torch.zeros(
            self.k + 1, self.V, dtype=dtype, device=self.device)
        self.in_q = torch.zeros(
            self.k, self.V, dtype=dtype, device=self.device)
        self.in_tokens = torch.zeros(
            self.k, dtype=torch.int64, device=self.device)

        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm):
            for _ in range(2):
                chain_proxy_candidates_fixed(
                    self.in_exit, self.in_q, self.in_tokens,
                    self.top_k, self.wire_n, self.pack_scores)
        warm.synchronize()
        torch.cuda.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out_pos, self.out_tok, self.out_piv = \
                chain_proxy_candidates_fixed(
                    self.in_exit, self.in_q, self.in_tokens,
                    self.top_k, self.wire_n, self.pack_scores)

    @torch.inference_mode()
    def replay(self, exit_logits, q_logits, tokens):
        if exit_logits.shape != self.in_exit.shape \
                or q_logits.shape != self.in_q.shape \
                or tokens.shape[0] < self.k:
            raise RuntimeError(
                "chain proxy graph input mismatch: "
                f"K={self.k} exit={tuple(exit_logits.shape)} "
                f"q={tuple(q_logits.shape)} tok={tuple(tokens.shape)}")
        self.in_exit.copy_(exit_logits)
        self.in_q.copy_(q_logits)
        self.in_tokens.copy_(tokens[:self.k])
        self.graph.replay()
        return self.out_pos, self.out_tok, self.out_piv


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
                    c_tensor: int, sampler_x=None, F=None,
                    assume_pos_temps: bool = False, generator=None,
                    noise=None):
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
    # gap-prof: GPU temps의 .any()→bool()은 forward 완료 대기 동기점
    # (2.4ms/forward). rollout은 진입 전 temp>0 확인 — 가드 생략 허용.
    if not assume_pos_temps and bool((temperatures <= 0).any()):
        raise ValueError(
            "tree_sample_wor: temperature==0 is gated (v6 §7.2 — "
            "support-exhaustion fallback intentionally not implemented; "
            "caller must fall back to the chain path)")
    probs = q_probs_from_logits(logits, temperatures, sampler_x, F)
    raw_q = probs.clone()                      # 원본 보존 (c_raw)
    epsilon = 1e-10
    # generator: P2 전용 CUDA graph-safe 제너레이터 (리뷰11-1 — 기본
    # 제너레이터는 P1/eager 전용으로 격리; None이면 종전 동작 그대로).
    # noise: 결정적 parity 전용 (리뷰12 §3) — [.,V] 고정 exponential
    # noise를 eager/graph 양쪽에 동일 주입해 비교 가능하게 함.
    if noise is not None:
        scores = probs.div_(noise + epsilon)
    else:
        scores = probs.div_(
            torch.empty_like(probs).exponential_(1, generator=generator)
            + epsilon)
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


@torch.inference_mode()
def warmup_tree_proxy_kernels(exit_logits: torch.Tensor,
                              q_logits: torch.Tensor,
                              par, sib, wire_n: int):
    """Run the exact production tree-proxy math without communication.

    Tree verification is sparse in normal traffic, so its shape-specific
    softmax/reduction/top-k kernels otherwise initialize on the first real P2
    hit.  That produced a 33 ms send-ready outlier even though steady hits
    were about 4.8 ms.  ModelRunner calls this once for every 4/6/8-node
    verify bucket during initialization, on the same side stream used by the
    live exit replica.

    This deliberately invokes the existing implementation rather than a
    reduced synthetic approximation: warmup must cover the topology H2D,
    sibling ladder, flattened candidate top-k, and P_iv packing kernels.
    """
    valid = len(par)
    if exit_logits.ndim != 2 or exit_logits.shape[0] != valid + 1:
        raise ValueError(
            "tree proxy warmup exit shape must be [valid+1,V]: "
            f"valid={valid} shape={tuple(exit_logits.shape)}")
    if q_logits.ndim != 2 or q_logits.shape[0] != valid:
        raise ValueError(
            "tree proxy warmup q shape must be [valid,V]: "
            f"valid={valid} shape={tuple(q_logits.shape)}")
    V = exit_logits.shape[-1]
    tokens = torch.arange(valid, dtype=torch.int64,
                          device=exit_logits.device) % V
    p_rows = torch.softmax(exit_logits.float(), dim=-1)
    q_rows = torch.softmax(q_logits.float(), dim=-1)
    _alpha, term, resid = tree_policy_b_ladder(
        par, sib, tokens, p_rows, q_rows)
    piv_rows = resid * term.unsqueeze(1)
    if valid:
        par_t = torch.tensor(par, dtype=torch.int64,
                             device=exit_logits.device)
        piv_rows[par_t + 1, tokens] = 0.0
    k = min(int(wire_n), piv_rows.numel())
    top_v, top_i = piv_rows.flatten().topk(k)
    chosen_tok = (top_i % V).to(torch.int64)
    pack_piv(chosen_tok, top_v)


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
                 depth_cap: int, tip_idx=None, root_remaining=None):
    """이번 forward에서 평가할 노드 선택 (D1 정책 스위치).

    level    : depth == fwd 인 미평가 노드만 (level-synchronous).
    frontier : 미평가 전체에서 priority 상위 W.
    공통: depth ≥ depth_cap 노드는 제외 (자식이 캡 초과라 확장 무의미).

    이슈 #27 (리뷰2-1): tip_idx/root_remaining이 주어지면 **잔여 예산이
    있는 root의 backbone tip은 의무 lane** — top-W 경합에서 탈락 불가.
    (탈락 시 level 정책은 그 depth를 재방문하지 않아 root의 깊이가
    영구 정지 + 예산 소실: budgets [7,7,7,7,6,6]·W10·F4·C3 재현에서
    40 중 34만 생성, 약root dmax=1.) 의무 tip 수 ≤ R ≤ W는 호출자
    (run_rollout의 R≤W 가드 + config root_count 검증)가 보장.
    반환: pool 인덱스 리스트 (≤ W; 부족하면 그만큼 — pad는 호출자).
    """
    n = pool.n
    elig = (pool.state[:n] == 0) & (pool.depth[:n] < depth_cap)
    # ``confidence`` is the canonical, low-knob policy.  Like EAGLE-2 it
    # advances level-synchronously and ranks candidates by cumulative path
    # confidence.  Legacy ``level`` remains as an exact compatibility name.
    if policy in (
            "level", "confidence", "coverage", "backbone", "eagle",
            "hybrid",
            "adaptive"):
        elig = elig & (pool.depth[:n] == fwd)
    idx = torch.nonzero(elig).flatten()
    if idx.numel() == 0:
        return []
    mand = []
    if tip_idx is not None and root_remaining is not None:
        elig_set = set(idx.tolist())
        for r, t in enumerate(tip_idx):
            if int(root_remaining[r]) > 0 and t in elig_set:
                mand.append(t)
    pri = pool.logpri[idx]
    order = torch.argsort(pri, descending=True, stable=True)
    ranked = idx[order].tolist()
    if not mand:
        return ranked[:W]
    mand_set = set(mand)
    rest = [i for i in ranked if i not in mand_set]
    # Root backbone rows keep their root order.  This is more than a
    # deterministic tie-break: with C=1 and R=W it makes the tree rollout
    # exactly degenerate to the established chain layout (root r remains in
    # lane r in every round).  Only surplus lanes are ranked by path score.
    # Reordering mandatory tips by score needlessly changed the physical KV
    # lane of every root on every round and broke that equivalence contract.
    return (mand + rest)[:W]


def select_nodes_global(pool: TreePool, W: int, fwd: int, depth_cap: int,
                        root_remaining: torch.Tensor,
                        future_rounds: int, proxy_threshold: float = 0.0,
                        conf_threshold: float = 0.0):
    """Select productive global-frontier parents for one fixed-width round.

    Plain top-W can waste lanes when many high-score candidates belong to a
    root whose final ``Nv`` view has only one slot left.  Rank globally, but
    admit at most the number of parents that can each produce one child this
    round.  This changes no probability law: it only decides which existing
    candidate nodes receive a later draft forward.
    """
    n = pool.n
    elig = ((pool.state[:n] == 0) & (pool.depth[:n] == fwd)
            & (pool.depth[:n] < depth_cap))
    if fwd > 0:
        # Floors control only *later expansion*.  The low-score node remains
        # in the response view and can still be accepted by the exact
        # verifier.  Round zero deliberately evaluates every live root.
        if proxy_threshold > 0.0:
            root_logp = pool.logpri[:n].gather(0, pool.root[:n])
            elig &= root_logp >= math.log(proxy_threshold)
        if conf_threshold > 0.0:
            elig &= pool.raw_q[:n] >= conf_threshold
    idx = torch.nonzero(elig).flatten()
    if idx.numel() == 0:
        return []
    order = torch.argsort(pool.logpri[idx], descending=True, stable=True)
    ranked = idx[order].tolist()
    future = max(0, int(future_rounds))
    rem = root_remaining.tolist()
    quota = [max(1, int(x) - future) if int(x) > 0 else 0 for x in rem]
    used = [0] * len(rem)
    out = []
    for i in ranked:
        r = int(pool.root[i])
        if used[r] >= quota[r]:
            continue
        out.append(i)
        used[r] += 1
        if len(out) == W:
            break
    return out


def _alloc_stats(pool: TreePool, budgets: torch.Tensor, R: int,
                 requested: int | None = None):
    """이슈 #27 불변 기록: requested/allocated/generated 3값 분리
    (리뷰3-7: cap(nv)이 requested를 먼저 자르므로 allocated만 보면
    이용률 손실이 안 보인다 — 예: W10·Nv4·R6은 requested 40 중
    allocated 24, 이용률 60%) + root별 dmax.
    """
    n = pool.n
    gen = [0] * R
    dmax = [0] * R
    for i in range(R, n):
        r = int(pool.root[i])
        gen[r] += 1
        d = int(pool.depth[i])
        if d > dmax[r]:
            dmax[r] = d
    alloc_l = budgets.tolist()
    return {"requested": requested, "allocated": int(sum(alloc_l)),
            "generated": int(n - R),
            "per_root_alloc": alloc_l, "per_root_gen": gen,
            "per_root_dmax": dmax}


def rollout_reference(root_toks, root_piv, root_pos, *, policy, W, F_total,
                      c_tensor, nv, beta, depth_cap, sample_fn,
                      fanout_policy="ctensor", proxy_threshold=0.0,
                      conf_threshold=0.0):
    """rollout 참조 구현 (T1.4a — 엔진 배선(T1.4b)의 정답지이자
    CPU 테스트 대상). sample_fn(node_indices, fanouts) -> (tokens [n, C],
    raw_q [n, C]) — 정체는 여기서만 관측 (D10: 예산·선택은 그 전에 확정).

    반환: (pool, eval_log) — eval_log[f] = (선택 인덱스, fanout) 기록.
    """
    R = len(root_toks)
    if policy in ("dynamic", "eagle", "hybrid"):
        # All roots are evaluated at depth zero.  Later levels compete
        # globally by cumulative root-proxy x path-draft confidence. Hybrid
        # keeps backbone handling only for its first two rounds.
        if policy in ("dynamic", "eagle"):
            fanout_policy = "ctensor"
        if R > W:
            raise ValueError(
                f"dynamic tree rollout requires R<=W so every root is "
                f"evaluated in round zero; got R={R}, W={W}")
    if fanout_policy == "backbone" and R > W:
        raise ValueError(
            f"tree rollout: root_count R={R} > W={W} — tip 의무 lane이 "
            f"W를 초과 (이슈 #27; backbone 정책은 R<=W 필요 — "
            f"duet_tree_root_count 설정 오류)")
    pool = TreePool(capacity=R + F_total * W * c_tensor)
    # The canonical confidence policy fixes the old beta sweep at the
    # validated square-root water filling.  The active-root count is derived
    # once by Config; legacy policies retain their explicit beta/R knobs for
    # exact experiment reproduction.
    budgets = alloc_policy_root_budgets(
        root_piv, policy, total=F_total * W, beta=beta, cap=nv)
    remaining = budgets.clone()
    logpiv = root_piv.clamp_min(1e-9).log()
    for r in range(R):
        pool.add(root_toks[r], -1, -1, 0, r, 0, float(logpiv[r]), 1.0)
    eval_log = []
    tip_idx = list(range(R))                      # backbone tip (root부터)
    hybrid_floor = min(2, F_total)
    for f in range(F_total):
        global_round = (policy in ("dynamic", "eagle")
                        or (policy == "hybrid" and f >= hybrid_floor))
        if global_round:
            sel = select_nodes_global(
                pool, W, f, depth_cap, remaining,
                future_rounds=F_total - f - 1,
                proxy_threshold=proxy_threshold,
                conf_threshold=conf_threshold)
        else:
            sel = select_nodes(
                pool, policy, W, f, depth_cap,
                tip_idx=tip_idx if fanout_policy == "backbone" else None,
                root_remaining=remaining if fanout_policy == "backbone"
                else None)
        if not sel:
            eval_log.append(([], None))
            continue
        pri = pool.logpri[torch.tensor(sel)]
        roots = pool.root[torch.tensor(sel)]
        if global_round:
            fan = alloc_fanouts_global(
                pri, roots, remaining, c_tensor,
                future_rounds=F_total - f - 1)
        elif fanout_policy == "backbone":
            is_tip = torch.tensor(
                [sel[k] == tip_idx[int(roots[k])] for k in range(len(sel))],
                dtype=torch.bool)
            if policy == "hybrid":
                fan = is_tip.long() * (remaining[roots] > 0).long()
            else:
                reserve = torch.tensor(
                    [max(0, depth_cap - int(pool.depth[tip_idx[r]]))
                     for r in range(R)], dtype=torch.int64)
                _fanout_fn = (alloc_fanouts_adaptive
                              if policy == "adaptive"
                              else alloc_fanouts_backbone)
                fan = _fanout_fn(pri, roots, remaining, reserve,
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
                if float(raws[k][c]) <= 0.0:      # WOR support 소진 (동상)
                    continue
                child = pool.add(int(toks[k][c]), i, cell,
                         int(pool.depth[i]) + 1, int(pool.root[i]), c,
                         float(pool.logpri[i])
                         + float(torch.log(torch.clamp(raws[k][c], min=1e-9))),
                         float(raws[k][c]))
                # backbone 연장: tip의 맏이(c=0)가 새 tip
                if not global_round and fanout_policy == "backbone" and c == 0 \
                        and i == tip_idx[int(pool.root[i])]:
                    tip_idx[int(pool.root[i])] = child
        eval_log.append((sel, fan))
    requested = (R * nv if policy in (
        "coverage", "backbone", "eagle", "hybrid", "adaptive")
                 else F_total * W)
    pool.alloc_stats = _alloc_stats(pool, budgets, R, requested=requested)
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
                F_x=None, pad_token=0, fanout_policy="ctensor",
                proxy_threshold=0.0, conf_threshold=0.0):
    """엔진-주입형 rollout (T1.4b-a): forward_fn만 바꾸면 stub/실엔진
    양쪽에서 동작. per-forward로 (input_ids[W], rope[W], packed mask)를
    구성해 forward_fn(f, input_ids, rope, packed, indptr) -> logits[W,V]
    를 호출하고, tree_sample_wor로 자식을 뽑아 pool을 채운다.

    rollout_reference와 topology가 동일해야 한다 (stub 테스트로 고정).
    pad 행: 유효 노드가 W 미만이면 pad_token/rope_base[0]로 채우고
    fanout 0 (자식 무시; RNG는 소비 — 고정 shape 유지).
    """
    R = len(root_toks)
    if policy in ("dynamic", "eagle", "hybrid"):
        if policy in ("dynamic", "eagle"):
            fanout_policy = "ctensor"
        if R > W:
            raise ValueError(
                f"dynamic tree rollout requires R<=W so every root is "
                f"evaluated in round zero; got R={R}, W={W}")
    if fanout_policy == "backbone" and R > W:
        raise ValueError(
            f"tree rollout: root_count R={R} > W={W} — tip 의무 lane이 "
            f"W를 초과 (이슈 #27; backbone 정책은 R<=W 필요 — "
            f"duet_tree_root_count 설정 오류)")
    pool = TreePool(capacity=R + F_total * W * c_tensor)
    budgets = alloc_policy_root_budgets(
        root_piv, policy, total=F_total * W, beta=beta, cap=nv)
    remaining = budgets.clone()
    logpiv = root_piv.clamp_min(1e-9).log()
    for r in range(R):
        pool.add(int(root_toks[r]), -1, -1, 0, r, 0, float(logpiv[r]), 1.0)
    eval_log = []
    cell_logits = None                     # [F·W, V] — verify q_eff 원천 (T2.2)
    tip_idx = list(range(R))               # backbone tip (root 자신부터)
    # gap-prof 슬림화: pool 텐서 캐스팅 대신 파이썬 미러 (결과 불변)
    tip_depth = [0] * R
    node_root = list(range(R))
    node_depth = [0] * R
    node_logpri = [float(x) for x in logpiv.tolist()]
    hybrid_floor = min(2, F_total)
    for f in range(F_total):
        global_round = (policy in ("dynamic", "eagle")
                        or (policy == "hybrid" and f >= hybrid_floor))
        if global_round:
            sel = select_nodes_global(
                pool, W, f, depth_cap, remaining,
                future_rounds=F_total - f - 1,
                proxy_threshold=proxy_threshold,
                conf_threshold=conf_threshold)
        else:
            sel = select_nodes(
                pool, policy, W, f, depth_cap,
                tip_idx=tip_idx if fanout_policy == "backbone" else None,
                root_remaining=remaining if fanout_policy == "backbone"
                else None)
        n_sel = len(sel)
        fan = torch.zeros(W, dtype=torch.int64)
        fan_l = []
        if n_sel:
            pri = torch.tensor([node_logpri[i] for i in sel])
            roots = torch.tensor([node_root[i] for i in sel])
            if global_round:
                fan[:n_sel] = alloc_fanouts_global(
                    pri, roots, remaining, c_tensor,
                    future_rounds=F_total - f - 1)
            elif fanout_policy == "backbone":
                is_tip = torch.tensor(
                    [sel[k] == tip_idx[node_root[sel[k]]]
                     for k in range(n_sel)], dtype=torch.bool)
                if policy == "hybrid":
                    fan[:n_sel] = (is_tip.long()
                                   * (remaining[roots] > 0).long())
                else:
                    reserve = torch.tensor(
                        [max(0, depth_cap - tip_depth[r]) for r in range(R)],
                        dtype=torch.int64)
                    _fanout_fn = (alloc_fanouts_adaptive
                                  if policy == "adaptive"
                                  else alloc_fanouts_backbone)
                    fan[:n_sel] = _fanout_fn(
                        pri, roots, remaining, reserve, is_tip, c_tensor)
            else:
                fan[:n_sel] = alloc_fanouts(pri, roots, remaining, c_tensor)
            fan_l = fan[:n_sel].tolist()
            for k in range(n_sel):
                remaining[node_root[sel[k]]] -= fan_l[k]
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
        _gp = os.environ.get("SSD_TREE_GAP_PROF", "0") == "1"
        if _gp:
            import time as _t
            _t0 = _t.perf_counter()
        packed, indptr = build_tree_mask_packed(
            f, W, K_glue, context_len, glue, anc, selfc)
        if _gp:
            _t1 = _t.perf_counter()
        logits = forward_fn(f, input_ids, rope, packed, indptr)
        if _gp:
            _t2 = _t.perf_counter()
        if cell_logits is None:
            # forward_fn 디바이스 상주 (엔진=GPU — [W,V] CPU 왕복 제거;
            # stub 테스트는 CPU 그대로)
            cell_logits = torch.zeros(F_total * W, logits.shape[-1],
                                      dtype=logits.dtype,
                                      device=logits.device)
        cell_logits[f * W:(f + 1) * W] = logits[:W]
        toks, raws = tree_sample_wor(logits, temps.to(logits.device),
                                     c_tensor, sampler_x=sampler_x, F=F_x,
                                     assume_pos_temps=True)
        if _gp:
            _t3 = _t.perf_counter()
        toks, raws = toks.cpu(), raws.cpu()   # pool 장부는 CPU (소량 1회)
        if _gp:
            _t4 = _t.perf_counter()
        toks_l = toks.tolist()
        raws_l = raws.tolist()
        for k in range(n_sel):
            i = sel[k]
            cell = f * W + k
            pool.cell[i] = cell
            pool.state[i] = 1
            for c in range(fan_l[k]):
                rq = raws_l[k][c]
                if rq <= 0.0:
                    # 리뷰3-10: WOR support 소진 — fan > nonzero support면
                    # zero-q 토큰이 top-k에 들어오고 verify 보행이
                    # D[t]==0 hard-fail. 분포 성질(support)만 본 사후
                    # 배제 (정체 무관 — D10 안전; 기존 .cpu() 편승,
                    # 추가 sync 0). 미소진 예산은 alloc_stats에 기록.
                    continue
                lp = node_logpri[i] + math.log(max(rq, 1e-9))
                child = pool.add(toks_l[k][c], i, cell,
                                 node_depth[i] + 1, node_root[i], c,
                                 lp, rq)
                node_root.append(node_root[i])
                node_depth.append(node_depth[i] + 1)
                node_logpri.append(lp)
                if not global_round and fanout_policy == "backbone" and c == 0 \
                        and i == tip_idx[node_root[i]]:
                    tip_idx[node_root[i]] = child
                    tip_depth[node_root[i]] = node_depth[i] + 1
        if _gp:
            _t5 = _t.perf_counter()
            print(f"[gap-prof] f={f} mask={( _t1-_t0)*1e3:.2f} "
                  f"fwd(plan+replay)={(_t2-_t1)*1e3:.2f} "
                  f"sample_launch={(_t3-_t2)*1e3:.2f} "
                  f"cpu_sync={(_t4-_t3)*1e3:.2f} "
                  f"pool={(_t5-_t4)*1e3:.2f}", flush=True)
        eval_log.append((sel, fan[:n_sel]))
    requested = (R * nv if policy in (
        "coverage", "backbone", "eagle", "hybrid", "adaptive")
                 else F_total * W)
    pool.alloc_stats = _alloc_stats(pool, budgets, R, requested=requested)
    if os.environ.get("SSD_TREE_ALLOC_CHECK", "0") == "1":
        st = pool.alloc_stats
        if st["generated"] != st["allocated"]:
            print(f"[tree-alloc #27] requested={st['requested']} "
                  f"generated={st['generated']} != "
                  f"allocated={st['allocated']} per_root_gen="
                  f"{st['per_root_gen']} alloc={st['per_root_alloc']} "
                  f"dmax={st['per_root_dmax']}", flush=True)
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
    # 1b (docs/duet/internal/22): ① 노드별 '텐서 인덱싱' 루프(~500 스칼라 op)를
    # 단일 tolist 후 파이썬 리스트 루프로 (수 ms → ~0.1ms); ② pq_logits
    # [R,nv,V] (~10MB) 선제 물질화 제거 — parent_q_cells만 만들고 서빙
    # 시 hit root 1개만 cell_logits에서 gather (리뷰3-12 권고).
    n = pool.n
    tok_l = pool.tok[:n].tolist()
    par_l = pool.parent_idx[:n].tolist()
    root_l = pool.root[:n].tolist()
    sib_l = pool.sib_order[:n].tolist()
    rq_l = pool.raw_q[:n].tolist()
    pc_l = pool.parent_cell[:n].tolist()
    tok = torch.zeros(R, nv, dtype=torch.int64)
    parent_local = torch.full((R, nv), -1, dtype=torch.int64)
    sib = torch.zeros(R, nv, dtype=torch.int64)
    raw_q = torch.zeros(R, nv)
    valid_l = [0] * R
    local_of = {}
    tk = [[0] * nv for _ in range(R)]
    pl = [[-1] * nv for _ in range(R)]
    sb = [[0] * nv for _ in range(R)]
    rq = [[0.0] * nv for _ in range(R)]
    pq_ref_l = [[-1] * nv for _ in range(R)]
    pq_cells_l = [[-1] * nv for _ in range(R)]
    uniq = [dict() for _ in range(R)]
    u_valid_l = [0] * R
    for i in range(n):
        p = par_l[i]
        if p < 0:
            continue                       # root 자체는 뷰에 안 들어감
        r = root_l[i]
        j = valid_l[r]
        if j >= nv:
            raise RuntimeError("생성-시점 캡 위반 (⑤v2) — view overflow "
                               f"root={r} j={j} nv={nv} (리뷰3: -O 생존 가드)")
        tk[r][j] = tok_l[i]
        pl[r][j] = local_of.get(p, -1)     # 부모가 root면 -1
        sb[r][j] = sib_l[i]
        rq[r][j] = rq_l[i]
        if cell_logits is not None:
            pc = pc_l[i]
            u = uniq[r].get(pc)
            if u is None:
                u = len(uniq[r])
                if u >= nv:
                    raise RuntimeError("U_max=N_v 위반")
                uniq[r][pc] = u
                pq_cells_l[r][u] = pc
                u_valid_l[r] = u + 1
            pq_ref_l[r][j] = u
        local_of[i] = j
        valid_l[r] = j + 1
    tok = torch.tensor(tk, dtype=torch.int64)
    parent_local = torch.tensor(pl, dtype=torch.int64)
    sib = torch.tensor(sb, dtype=torch.int64)
    raw_q = torch.tensor(rq)
    valid = torch.tensor(valid_l, dtype=torch.int64)
    out = {"tok": tok, "parent_local": parent_local, "sib_order": sib,
           "raw_q": raw_q, "valid": valid}
    if cell_logits is not None:
        out["parent_q_ref"] = torch.tensor(pq_ref_l, dtype=torch.int64)
        out["parent_q_cells"] = torch.tensor(pq_cells_l,
                                             dtype=torch.int64)
        out["u_valid"] = torch.tensor(u_valid_l, dtype=torch.int64)
        out["cell_logits"] = cell_logits    # 서빙-시 hit root만 gather
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
    """Pack one root into a common fixed-width topology sidecar.

    ``view`` may have a smaller phase-local width than ``nv`` (for example,
    P2 has eight nodes while the shared P1/P2 wire has thirteen).  The live
    prefix is copied and the remainder stays zero-padded.  This keeps one
    collective shape without forcing either executor to allocate the other
    phase's larger output tensors.
    """
    dev = (view["valid"].device
           if torch.is_tensor(view["valid"]) else torch.device("cpu"))
    out = torch.zeros(tree_wire_ints_len(nv), dtype=torch.int64,
                      device=dev)
    if hit_root < 0:
        return out
    r = hit_root
    view_nv = int(view["tok"].shape[1])
    copy_n = min(view_nv, int(nv))
    valid = view["valid"][r]
    if __debug__ and int(valid) > copy_n:
        raise RuntimeError(
            f"tree view valid={int(valid)} exceeds pack width {copy_n}")
    out[0] = valid
    out[1] = view["u_valid"][r]
    out[2] = 1                                   # epoch/버전 자리 (T2.4)
    o = 3
    out[o:o + copy_n] = view["tok"][r, :copy_n]
    out[o + nv:o + nv + copy_n] = \
        view["parent_local"][r, :copy_n]
    out[o + 2 * nv:o + 2 * nv + copy_n] = \
        view["sib_order"][r, :copy_n]
    out[o + 3 * nv:o + 3 * nv + copy_n] = \
        view["parent_q_ref"][r, :copy_n]
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


def validate_tree_ints(tree_ints, nv: int, vocab_size: int | None = None):
    """Validate the topology before any target collective or tree walk.

    A parent must precede its child in generation order.  Without this
    invariant a malformed self/cycle can make the rank-0 acceptance walk loop
    forever after the other tensor-parallel ranks have completed, producing
    the characteristic GPU0-only 100% NCCL spin.  Keep this small CPU check at
    the wire boundary where the metadata is already being read back.
    """
    valid = int(tree_ints["valid"])
    u_valid = int(tree_ints["u_valid"])
    if valid < 0 or valid > nv:
        raise RuntimeError(f"tree valid={valid} outside [0,{nv}]")
    if u_valid < 0 or u_valid > valid:
        raise RuntimeError(
            f"tree u_valid={u_valid} outside [0,valid={valid}]")
    children = {}
    for j in range(valid):
        parent = int(tree_ints["parent_local"][j])
        sibling = int(tree_ints["sib_order"][j])
        qref = int(tree_ints["parent_q_ref"][j])
        token = int(tree_ints["tok"][j])
        if parent < -1 or parent >= j:
            raise RuntimeError(
                f"tree parent invariant failed at node {j}: parent={parent} "
                f"(expected -1 <= parent < {j})")
        if sibling < 0:
            raise RuntimeError(
                f"tree sibling order is negative at node {j}: {sibling}")
        if qref < 0 or qref >= u_valid:
            raise RuntimeError(
                f"tree parent-q ref failed at node {j}: qref={qref}, "
                f"u_valid={u_valid}")
        if vocab_size is not None and not (0 <= token < vocab_size):
            raise RuntimeError(
                f"tree token out of range at node {j}: token={token}, "
                f"vocab={vocab_size}")
        children.setdefault(parent, []).append((sibling, qref))
    for parent, entries in children.items():
        sibs = sorted(s for s, _ in entries)
        if sibs != list(range(len(entries))):
            raise RuntimeError(
                f"tree sibling order for parent {parent} is {sibs}, "
                f"expected {list(range(len(entries)))}")
        qrefs = {q for _, q in entries}
        if len(qrefs) != 1:
            raise RuntimeError(
                f"tree siblings for parent {parent} use multiple q refs: "
                f"{sorted(qrefs)}")
    return tree_ints


def tree_parent_path(tree_ints, terminal_node: int):
    """Return root-to-terminal local node indices with a hard bound.

    ``terminal_node`` uses DUET's wire namespace: 0 means the root context
    and 1+j means view node j.  This helper deliberately works on CPU tree
    metadata; reconstructing an at-most-eight-node path must never issue a
    sequence of scalar CUDA synchronizations in the draft request loop.
    """
    valid = int(tree_ints["valid"])
    terminal_node = int(terminal_node)
    if terminal_node == 0:
        return []
    if terminal_node < 0 or terminal_node > valid:
        raise RuntimeError(
            f"tree terminal node {terminal_node} outside [0,{valid}]")
    parent = tree_ints["parent_local"]
    path = []
    node = terminal_node - 1
    for _ in range(valid):
        path.append(node)
        next_node = int(parent[node])
        if next_node < 0:
            path.reverse()
            return path
        if next_node >= node:
            raise RuntimeError(
                f"tree parent path is not strictly decreasing: "
                f"node={node}, parent={next_node}")
        node = next_node
    raise RuntimeError(
        f"tree parent path exceeded valid-node bound ({valid})")


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
    valid = int(tree_ints["valid"])
    # Use the physical row capacity, not ``valid`` itself, so corrupt wire
    # headers are rejected as well as corrupt parent links.
    _checked_tree = tree_ints
    if "u_valid" not in _checked_tree:
        # Pure/reference callers historically supplied only the fields the
        # walk consumes.  The production wire always carries u_valid; infer
        # it from the actual q table for backwards-compatible unit use.
        _checked_tree = dict(tree_ints)
        _checked_tree["u_valid"] = int(q_parent_probs.shape[0])
    _tok_rows = tree_ints["tok"]
    _capacity = (int(_tok_rows.numel()) if torch.is_tensor(_tok_rows)
                 else len(_tok_rows))
    validate_tree_ints(_checked_tree, _capacity)
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
    visited = set()
    ctx = -1
    # A valid walk accepts at most one new node per iteration.  Keep the
    # explicit bound even though validate_tree_ints() enforces parent<child:
    # a future change to either routine must fail instead of leaving rank 0
    # spinning while the other tensor-parallel ranks wait in a collective.
    for _ in range(valid + 1):
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
        if accepted in visited:
            raise RuntimeError(
                f"tree acceptance walk revisited node {accepted}")
        visited.add(accepted)
        path.append(accepted)
        ctx = accepted
    raise RuntimeError(
        f"tree acceptance walk exceeded valid-node bound ({valid})")


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


# ====================================================================
# T6 1a — GPU 상주 rollout (docs/duet/internal/22 v2). 정책·산술·RNG 소비 순서를
# run_rollout과 동일하게 유지한 채, forward 사이 CPU readback을 0회로.
# 동등성 게이트: tests의 arena-vs-CPU 라운드 트레이스 비교로 고정.
# ====================================================================

def alloc_root_budgets_gpu(piv: torch.Tensor, total: int, beta: float,
                           cap: int) -> torch.Tensor:
    """alloc_root_budgets의 무동기(sync-free) 미러 — float64 동일 산술,
    고정 반복(브레이크 대신 no-op 수렴), .item() 0회. CPU/GPU 텐서 모두
    동작하며 값은 CPU판과 정확히 일치 (동등성 테스트로 고정)."""
    R = piv.numel()
    dev = piv.device
    elig = piv > 0
    w = torch.where(elig, piv.clamp_min(1e-9).double().pow(beta),
                    torch.zeros(R, dtype=torch.float64, device=dev))
    quota = torch.zeros(R, dtype=torch.float64, device=dev)
    active = elig.clone()
    # 캡처 호환 (리뷰9-6/PoC 실측): torch.tensor(스칼라, device)는
    # pageable H2D — capture 중 금지. full()은 fill 커널이라 안전.
    left = torch.full((), float(total), dtype=torch.float64, device=dev)
    zero = torch.zeros((), dtype=torch.float64, device=dev)
    cap_f = torch.full((), float(cap), dtype=torch.float64, device=dev)
    for _ in range(R):
        wa = torch.where(active, w, torch.zeros_like(w))
        add = wa / wa.sum().clamp_min(1e-300) * left
        newq = quota + add
        over = active & (newq >= cap)
        anyover = over.any()
        take = torch.where(over, cap_f - quota, torch.zeros_like(quota))
        quota = torch.where(anyover,
                            torch.where(over, cap_f.expand(R), quota),
                            newq)
        left = torch.where(anyover, left - take.sum(), zero)
        active = active & ~over
    base = quota.floor().long().clamp_max(cap)
    rem = (total - base.sum()).clamp(min=0)
    rem = torch.minimum(rem, elig.long().sum() * cap - base.sum())
    frac = quota - quota.floor()
    order = torch.argsort(frac, descending=True, stable=True)
    for _ in range(2):                     # largest-remainder, 고정 2패스
        ok = (base < cap) & elig
        ok_o = ok.index_select(0, order)
        cum = ok_o.long().cumsum(0)
        take_o = ok_o & (cum <= rem)
        base.scatter_add_(0, order, take_o.long())
        rem = rem - take_o.long().sum()
    return base


_ANC_WORD_BITS = 63


def ancestry_word_count(max_cells: int) -> int:
    """Number of signed-int64 words needed for forward-cell ancestry.

    Bit 63 is deliberately unused so PyTorch/Triton signed right shifts keep
    their previous logical meaning.  The old P2 shape (40 cells) remains one
    word; wider P1 shapes transparently use more words.
    """
    if max_cells < 1:
        return 1
    return (int(max_cells) + _ANC_WORD_BITS - 1) // _ANC_WORD_BITS


def _arena_get(capacity, dev, workspace=None, max_cells=63):
    """persistent arena (리뷰6 §6): rollout마다 ~15개 텐서 신규 할당
    대신 workspace dict에서 재사용 + reset. workspace=None이면 신규
    (테스트 경로)."""
    if workspace is None:
        return TreeArena(capacity, dev, max_cells=max_cells)
    nwords = ancestry_word_count(max_cells)
    key = (capacity, str(dev), nwords)
    ar = workspace.get(key)
    if ar is None:
        ar = TreeArena(capacity, dev, max_cells=max_cells)
        workspace[key] = ar
    else:
        ar.reset()
    return ar


class TreeArena:
    """rollout 상태 전체를 device 텐서로 (22번 v2 — state 필드 명시,
    조상 bitset은 signed-int64 63-bit word 여러 개로 저장한다. P2의
    F·W=40은 기존과 동일한 1-word hot path, 더 넓은 P1만 multi-word)."""

    def __init__(self, capacity: int, device, max_cells: int = 63):
        d = device
        capacity = capacity + 1     # 말단 scratch 슬롯 (무효 쓰기 흡수
        #                             — boolean indexing의 nonzero/DtoH
        #                             동기화 회피, 리뷰5)
        self.tok = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.parent_idx = torch.full((capacity,), -1, dtype=torch.int64,
                                     device=d)
        self.parent_cell = torch.full((capacity,), -1, dtype=torch.int64,
                                      device=d)
        self.depth = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.root = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.sib = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.logpri = torch.full((capacity,), float("-inf"),
                                 dtype=torch.float64, device=d)
        self.raw_q = torch.ones(capacity, dtype=torch.float64, device=d)
        self.state = torch.zeros(capacity, dtype=torch.int64, device=d)
        self.cell = torch.full((capacity,), -1, dtype=torch.int64,
                               device=d)
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=d)
        self.anc_words = ancestry_word_count(max_cells)
        self.max_cells = int(max_cells)
        self.anc_bits = torch.zeros(
            capacity, self.anc_words, dtype=torch.int64, device=d)
        self.n = torch.zeros((), dtype=torch.int64, device=d)
        self.capacity = capacity
        self.device = d

    def reset(self):
        """persistent 재사용용 초기화 — 이전 rollout 상태 소거."""
        self.parent_idx.fill_(-1)
        self.parent_cell.fill_(-1)
        self.logpri.fill_(float("-inf"))
        self.state.zero_()
        self.cell.fill_(-1)
        self.valid.zero_()
        self.anc_bits.zero_()
        self.n.zero_()
        # tok/depth/root/sib/raw_q는 삽입 시 전량 덮어씀 (scratch 제외
        # — scratch 슬롯 값은 정의상 미사용)

    def to_pool(self, R: int):
        """단일 sync: CPU TreePool로 실체화 (+#38 무효 슬롯 압축 재매김
        — CPU run_rollout과 동일한 인덱스 공간·순서 보장). 전 과정
        텐서 압축 (파이썬 노드 루프 금지 — v1 루프가 build→merge를
        +16ms 부풀린 실측 교훈)."""
        n = int(self.n)                     # sync ① (스칼라)
        ints = torch.stack([self.tok[:n], self.parent_idx[:n],
                            self.depth[:n], self.root[:n],
                            self.sib[:n], self.state[:n],
                            self.cell[:n],
                            self.valid[:n].long()]).cpu()   # sync ②
        flts = torch.stack([self.logpri[:n],
                            self.raw_q[:n]]).cpu()          # sync ③
        keep = ints[7].bool()
        kept = torch.nonzero(keep).flatten()                # CPU — sync 無
        m = int(kept.numel())
        remap = torch.full((n + 1,), -1, dtype=torch.int64)
        remap[kept] = torch.arange(m)
        par_old = ints[1].index_select(0, kept)
        par_new = torch.where(par_old >= 0,
                              remap.gather(0, par_old.clamp(min=0)),
                              par_old)
        pcell = torch.where(par_old >= 0,
                            ints[6].gather(0, par_old.clamp(min=0)),
                            torch.full_like(par_old, -1))
        pool = TreePool(capacity=self.capacity)
        pool.n = m
        pool.tok[:m] = ints[0].index_select(0, kept)
        pool.parent_idx[:m] = par_new
        pool.parent_cell[:m] = pcell
        pool.depth[:m] = ints[2].index_select(0, kept)
        pool.root[:m] = ints[3].index_select(0, kept)
        pool.sib_order[:m] = ints[4].index_select(0, kept)
        pool.logpri[:m] = flts[0].index_select(0, kept).float()
        pool.raw_q[:m] = flts[1].index_select(0, kept).float()
        pool.state[:m] = ints[5].index_select(0, kept)
        pool.cell[:m] = ints[6].index_select(0, kept)
        if hasattr(self, "_budgets") and \
                os.environ.get("SSD_TREE_ALLOC_CHECK", "0") == "1":
            # 4번째 DtoH — 진단 게이트에서만 (리뷰6: "3 sync" 정정)
            pool.alloc_stats = _alloc_stats(
                pool, self._budgets.cpu(), R,
                requested=getattr(self, "_requested", None))
        return pool


def _arena_select(ar: TreeArena, policy, W, f, depth_cap, tip_idx,
                  remaining):
    """select_nodes 미러 (무동기): 반환 sel [W] (pad는 임의 slot),
    sel_valid [W]. 의무 tip 우선 + priority 순 — CPU와 동일 규약."""
    cap = ar.capacity
    dev = ar.device
    idxs = torch.arange(cap, device=dev)
    elig = (idxs < ar.n) & (ar.state == 0) & (ar.depth < depth_cap) \
        & ar.valid
    if policy in ("level", "confidence", "coverage", "backbone"):
        elig = elig & (ar.depth == f)
    # One sort, with two explicit groups:
    #   1) mandatory backbone tips in root order (root 0, 1, ...),
    #   2) surplus candidates in descending path score.
    # The first rule preserves exact chain lane/KV placement at C=1,R=W;
    # the second keeps dynamic score-based branching for spare lanes.
    mand_slot = torch.zeros(cap, dtype=torch.bool, device=dev)
    if tip_idx is not None:
        t = tip_idx.clamp(min=0)
        t_ok = (tip_idx >= 0) & (remaining > 0) & elig.gather(0, t)
        mand_slot.scatter_(0, t, t_ok)
    base = ar.logpri.float().double()      # CPU 비교 정밀도(f32) 고정
    mandatory_key = 1000.0 - ar.root.double()
    key = torch.where(
        elig,
        torch.where(mand_slot, mandatory_key, base),
        torch.full_like(base, float("-inf")))
    final = torch.argsort(key, descending=True, stable=True)
    elig_o = elig.gather(0, final)
    sel = final[:W]
    sel_valid = elig_o[:W]
    return sel, sel_valid


def _arena_fanout_backbone(ar: TreeArena, sel, sel_valid, tip_idx,
                           remaining, reserve, c_tensor, R):
    """alloc_fanouts_backbone 미러 (무동기). 반환 fan [W] (호출자가
    remaining 차감 — CPU 경로와 동일 분리)."""
    dev = ar.device
    W = sel.shape[0]
    r_of = ar.root.gather(0, sel.clamp(min=0))
    r_of = torch.where(sel_valid, r_of, torch.zeros_like(r_of))
    out = torch.zeros(W, dtype=torch.int64, device=dev)
    rem = remaining.clone()
    res = reserve.clone()
    # phase 1: tip fan 1 (같은 root의 tip은 유일)
    is_tip = sel_valid & (sel == tip_idx.gather(0, r_of))
    tip_take = is_tip & (rem.gather(0, r_of) > 0)
    out = out + tip_take.long()
    rem.scatter_add_(0, r_of, -tip_take.long())
    res_dec = torch.zeros_like(res)
    res_dec.scatter_add_(0, r_of, tip_take.long())
    res = (res - res_dec).clamp(min=0)
    # phase 2: priority 내림차순 라운드 (+1씩, c_tensor 라운드 = 정확 상계)
    pri = torch.where(sel_valid, ar.logpri.gather(0, sel.clamp(min=0)),
                      torch.full((W,), float("-inf"), dtype=torch.float64,
                                 device=dev)).float()
    lane_order = torch.argsort(pri, descending=True, stable=True)
    r_sorted = r_of.gather(0, lane_order)
    onehot = torch.nn.functional.one_hot(r_sorted, R).long()   # [W, R]
    for _ in range(c_tensor):
        uncapped = (out.gather(0, lane_order) < c_tensor) \
            & sel_valid.gather(0, lane_order)
        oh = onehot * uncapped.unsqueeze(1).long()
        rank = oh.cumsum(0) - oh                                # 앞행 수
        rank_of_lane = (rank * oh).sum(1)
        avail = (rem - res).clamp(min=0).gather(0, r_sorted)
        take = uncapped & (rank_of_lane < avail)
        out.scatter_add_(0, lane_order, take.long())
        rem = rem - (oh * take.unsqueeze(1).long()).sum(0)
    return out


def _arena_fanout_adaptive(ar: TreeArena, sel, sel_valid, tip_idx,
                           remaining, reserve, c_tensor, R):
    """GPU mirror of :func:`alloc_fanouts_adaptive` (fixed shape)."""
    out = _arena_fanout_backbone(
        ar, sel, sel_valid, tip_idx, remaining, reserve, c_tensor, R)
    r_of = torch.where(
        sel_valid, ar.root.gather(0, sel.clamp(min=0)),
        torch.zeros_like(sel))
    is_tip = sel_valid & (sel == tip_idx.gather(0, r_of))
    pri = torch.where(
        sel_valid, ar.logpri.gather(0, sel.clamp(min=0)),
        torch.full((sel.shape[0],), float("-inf"), dtype=torch.float64,
                   device=ar.device)).float()
    live = is_tip & torch.isfinite(pri)
    n_live = live.long().sum().clamp(min=1).float()
    masked = torch.where(live, pri, torch.full_like(pri, float("-inf")))
    threshold = torch.logsumexp(masked, 0) - torch.log(n_live)
    extra_ok = sel_valid & torch.isfinite(pri) & (pri >= threshold)
    floor = is_tip.long()
    return torch.where(extra_ok, out, torch.minimum(out, floor))


def _arena_select_global(ar: TreeArena, W, f, depth_cap, remaining,
                         future_rounds, R, proxy_threshold=0.0,
                         conf_threshold=0.0):
    """Fixed-shape GPU mirror of :func:`select_nodes_global`."""
    cap = ar.capacity
    dev = ar.device
    idxs = torch.arange(cap, device=dev)
    elig = ((idxs < ar.n) & (ar.state == 0) & ar.valid
            & (ar.depth == f) & (ar.depth < depth_cap))
    if f > 0:
        if proxy_threshold > 0.0:
            root_logp = ar.logpri.gather(0, ar.root.clamp(min=0))
            elig &= root_logp >= math.log(proxy_threshold)
        if conf_threshold > 0.0:
            elig &= ar.raw_q >= conf_threshold
    base = torch.where(
        elig, ar.logpri,
        torch.full_like(ar.logpri, float("-inf")))
    ranked = torch.argsort(base, descending=True, stable=True)
    elig_sorted = elig.gather(0, ranked)
    roots = ar.root.gather(0, ranked.clamp(min=0))
    oh = torch.nn.functional.one_hot(roots, R).long() \
        * elig_sorted.unsqueeze(1).long()
    ordinal = ((oh.cumsum(0) - oh) * oh).sum(1)
    future = max(0, int(future_rounds))
    quota = torch.where(
        remaining > 0,
        torch.maximum(torch.ones_like(remaining), remaining - future),
        torch.zeros_like(remaining))
    keep = elig_sorted & (ordinal < quota.gather(0, roots))
    key = torch.where(
        keep, base.gather(0, ranked),
        torch.full_like(base, float("-inf")))
    order2 = torch.argsort(key, descending=True, stable=True)
    sel = ranked.gather(0, order2)[:W]
    sel_valid = keep.gather(0, order2)[:W]
    return sel, sel_valid


def _arena_fanout_global(ar: TreeArena, sel, sel_valid, remaining,
                         c_tensor, R, future_rounds):
    """GPU mirror of :func:`alloc_fanouts` for global expansion.

    Selected parents are processed by cumulative path confidence.  Each may
    retain up to ``c_tensor`` already-sampled children, but parents belonging
    to the same root share that root's remaining response capacity.  This is
    fixed-shape and contains no host readback, so it is safe inside the P2
    CUDA graph.
    """
    dev = ar.device
    W = sel.shape[0]
    r_of = ar.root.gather(0, sel.clamp(min=0))
    r_of = torch.where(sel_valid, r_of, torch.zeros_like(r_of))
    pri = torch.where(
        sel_valid, ar.logpri.gather(0, sel.clamp(min=0)),
        torch.full((W,), float("-inf"), dtype=torch.float64,
                   device=dev)).float()
    lane_order = torch.argsort(pri, descending=True, stable=True)
    r_sorted = r_of.gather(0, lane_order)
    onehot = torch.nn.functional.one_hot(r_sorted, R).long()
    out = torch.zeros(W, dtype=torch.int64, device=dev)
    future = max(0, int(future_rounds))
    now = torch.where(
        remaining > 0,
        torch.maximum(torch.ones_like(remaining), remaining - future),
        torch.zeros_like(remaining))
    for _ in range(c_tensor):
        # First pass gives every selected parent one child.  Later passes add
        # siblings in the same global-confidence order.
        uncapped = ((out.gather(0, lane_order) < c_tensor)
                    & sel_valid.gather(0, lane_order))
        oh = onehot * uncapped.unsqueeze(1).long()
        rank = oh.cumsum(0) - oh
        rank_of_lane = (rank * oh).sum(1)
        avail = now.gather(0, r_sorted)
        take = uncapped & (rank_of_lane < avail)
        out.scatter_add_(0, lane_order, take.long())
        now = now - (oh * take.unsqueeze(1).long()).sum(0)
    return out


def _arena_mask_pack(f, W, K_glue, context_len, glue_sel, anc_sel,
                     sel_valid, device):
    """build_tree_mask_packed 미러 (GPU packbits little). 반환
    (packed uint8 [ceil(W·cols/8)], indptr int32 [2]) — device 상주."""
    cols = int(context_len) + f * W
    ttl = (f + 1) * W + (K_glue + 1)
    prefix_len = cols - ttl
    spec_w = (f + 1) * W
    m = torch.zeros(W, cols, dtype=torch.uint8, device=device)
    m[:, :prefix_len] = 1
    m[:, prefix_len:prefix_len + K_glue + 1] = \
        glue_sel * sel_valid.unsqueeze(1).to(torch.uint8)
    spec0 = prefix_len + K_glue + 1
    shifts = torch.arange(spec_w, device=device)
    word = torch.div(shifts, _ANC_WORD_BITS, rounding_mode="floor")
    bit = shifts.remainder(_ANC_WORD_BITS)
    anc_word = anc_sel.index_select(1, word)
    bits = ((anc_word >> bit.unsqueeze(0)) & 1).to(torch.uint8)
    selfbit = torch.zeros(W, spec_w, dtype=torch.uint8, device=device)
    lane = torch.arange(W, device=device)
    selfbit[lane, f * W + lane] = 1
    m[:, spec0:] = (bits | selfbit) * sel_valid.unsqueeze(1).to(torch.uint8)
    flat = m.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8,
                                            device=device)])
    weights = (1 << torch.arange(8, device=device)).to(torch.uint8)
    packed = (flat.view(-1, 8) * weights).sum(1, dtype=torch.int64) \
        .to(torch.uint8)
    indptr = torch.tensor([0, packed.numel()], dtype=torch.int32,
                          device=device)
    return packed, indptr


def run_rollout_arena(root_toks, root_piv, *, policy, W, F_total,
                      c_tensor, nv, beta, depth_cap, temps, forward_fn,
                      glue_rows_by_root, rope_base_by_root, K_glue,
                      context_len, sampler_x=None, F_x=None, pad_token=0,
                      fanout_policy="backbone", device=None,
                      workspace=None, p2_gen=None, noise_list=None,
                      trace_out=None, proxy_threshold=0.0,
                      conf_threshold=0.0):
    """run_rollout의 GPU 상주판 (T6 1a — 22번 v2 1단계).

    정책·예산 산술(float64)·선택/fanout 규약·RNG 소비 순서([W,V]
    tree_sample_wor per forward)를 CPU판과 동일하게 유지 — 차이는
    계산 장소뿐. forward 루프 내 CPU readback 0회; 유일한 sync는
    반환 후 to_pool() (1a 임시 — 1b에서 view/wire GPU화로 제거).

    Args (run_rollout 대비): root_piv/temps는 device 텐서 권장 (CPU면
    올림); glue_rows_by_root [R, K_glue+1] uint8, rope_base_by_root
    [R] int64 (텐서/리스트 허용). forward_fn은 device 텐서를 받는다.

    Returns: (arena, eval_trace, cell_logits) — eval_trace =
    (sel [F,W], sel_valid [F,W], fan [F,W]) device 텐서.
    """
    R = len(root_toks)
    if policy in ("dynamic", "eagle", "hybrid"):
        if policy in ("dynamic", "eagle"):
            fanout_policy = "ctensor"
        if R > W:
            raise ValueError(
                f"dynamic tree rollout requires R<=W so every root is "
                f"evaluated in round zero; got R={R}, W={W}")
    elif fanout_policy != "backbone":
        raise NotImplementedError(
            "arena supports only backbone/dynamic global policies (T6 1a)")
    if R > W:
        raise ValueError(f"tree rollout: R={R} > W={W} (이슈 #27)")
    dev = torch.device(device) if device is not None \
        else (root_piv.device if torch.is_tensor(root_piv) else "cpu")
    piv = (root_piv if torch.is_tensor(root_piv)
           else torch.tensor(root_piv)).to(dev)
    toks0 = (root_toks if torch.is_tensor(root_toks)
             else torch.tensor(root_toks, dtype=torch.int64)).to(dev)
    glue_rows = (glue_rows_by_root if torch.is_tensor(glue_rows_by_root)
                 else torch.as_tensor(glue_rows_by_root)) \
        .to(device=dev, dtype=torch.uint8)
    rope_base = (rope_base_by_root if torch.is_tensor(rope_base_by_root)
                 else torch.tensor(rope_base_by_root,
                                   dtype=torch.int64)).to(dev)
    temps_dev = temps.to(dev)

    ar = _arena_get(
        R + F_total * W * c_tensor, dev, workspace,
        max_cells=F_total * W)
    # 예산: CPU 정확판 + 1 sync (교대 A/B 실측 — GPU 무동기판은 ~256
    # 이벤트로 pre +3.4ms의 주범; piv.cpu()는 proxy_wait 직후라 큐가
    # 얕아 ~0.4ms. 패리티는 CPU 함수 그 자체이므로 자명)
    budgets = alloc_policy_root_budgets(
        piv.cpu(), policy, total=F_total * W, beta=beta, cap=nv).to(
            dev, non_blocking=True)
    ar._budgets = budgets
    ar._requested = (R * nv if policy in (
        "coverage", "backbone", "dynamic", "eagle", "hybrid", "adaptive")
                     else F_total * W)
    remaining = budgets.clone()
    # CPU 경로와 동일 정밀도: f32 log 후 double 확장 (리뷰6 —
    # double-log는 near-tie에서 1ULP 차이로 선택 순서를 바꿀 수 있음)
    logpiv = piv.clamp_min(1e-9).float().log().double()
    aR = torch.arange(R, device=dev)
    ar.tok[:R] = toks0
    ar.root[:R] = aR
    ar.logpri[:R] = logpiv
    # Physical roots remain present so CPU/arena node indices stay identical.
    # Zero P_iv means zero child budget; sanitize_root_inputs guarantees that
    # their fixed-width padding forwards still use valid token/rope values.
    ar.valid[:R] = True
    ar.n += R
    tip_idx = aR.clone()
    tip_depth = torch.zeros(R, dtype=torch.int64, device=dev)
    sel_tr, val_tr, fan_tr = [], [], []
    cell_logits = None
    hybrid_floor = min(2, F_total)
    for f in range(F_total):
        global_round = (policy in ("dynamic", "eagle")
                        or (policy == "hybrid" and f >= hybrid_floor))
        _tips = None if global_round else tip_idx
        if global_round:
            sel, sel_valid = _arena_select_global(
                ar, W, f, depth_cap, remaining,
                future_rounds=F_total - f - 1, R=R,
                proxy_threshold=proxy_threshold,
                conf_threshold=conf_threshold)
            fan = _arena_fanout_global(
                ar, sel, sel_valid, remaining, c_tensor, R,
                future_rounds=F_total - f - 1)
        else:
            sel, sel_valid = _arena_select(ar, policy, W, f, depth_cap,
                                           _tips, remaining)
            if policy == "hybrid":
                r_sel = ar.root.gather(0, sel.clamp(min=0))
                is_tip = sel_valid & (sel == tip_idx.gather(0, r_sel))
                fan = (is_tip & (remaining.gather(0, r_sel) > 0)).long()
            else:
                reserve = (depth_cap - tip_depth).clamp(min=0)
                _fanout_fn = (_arena_fanout_adaptive
                              if policy == "adaptive"
                              else _arena_fanout_backbone)
                fan = _fanout_fn(
                    ar, sel, sel_valid, tip_idx, remaining, reserve,
                    c_tensor, R)
        r_of = torch.where(
            sel_valid, ar.root.gather(0, sel.clamp(min=0)),
            torch.zeros_like(sel))
        remaining.scatter_add_(0, r_of, -fan)
        # --- forward 입력 (전부 device) ---
        input_ids = torch.where(sel_valid,
                                ar.tok.gather(0, sel.clamp(min=0)),
                                torch.full_like(sel, pad_token))
        if os.environ.get("SSD_CG_INPUT_CHECK", "0") == "1" \
                and bool((input_ids < 0).any()):
            print(f"[neg-ids-src] f={f} ids={input_ids.tolist()} "
                  f"sel={sel.tolist()} valid={sel_valid.tolist()} "
                  f"tok0_R={ar.tok[:R].tolist()} "
                  f"toks0_in={toks0.tolist()}", flush=True)
        rope = torch.where(
            sel_valid,
            rope_base.gather(0, r_of) + ar.depth.gather(0, sel.clamp(min=0)),
            rope_base[0].expand(W))
        glue_sel = glue_rows.index_select(0, r_of)
        anc_sel = ar.anc_bits.index_select(0, sel.clamp(min=0)) \
            * sel_valid.long().unsqueeze(1)
        packed, indptr = _arena_mask_pack(
            f, W, K_glue, context_len, glue_sel, anc_sel, sel_valid, dev)
        logits = forward_fn(f, input_ids, rope, packed, indptr)
        if cell_logits is None:
            cell_logits = torch.zeros(F_total * W, logits.shape[-1],
                                      dtype=logits.dtype,
                                      device=logits.device)
        cell_logits[f * W:(f + 1) * W] = logits[:W]
        sample_kwargs = dict(
            sampler_x=sampler_x, F=F_x, assume_pos_temps=True)
        if p2_gen is not None:
            sample_kwargs["generator"] = p2_gen
        if noise_list is not None:
            sample_kwargs["noise"] = noise_list[f]
        toks, raws = tree_sample_wor(
            logits, temps_dev, c_tensor, **sample_kwargs)
        if trace_out is not None:
            # 단계1 진단: forward별 아티팩트 (호출자 소유 dict)
            trace_out.setdefault("ids", []).append(input_ids.clone())
            trace_out.setdefault("rope", []).append(rope.clone())
            trace_out.setdefault("packed", []).append(packed.clone())
            trace_out.setdefault("logits", []).append(logits.clone())
            trace_out.setdefault("toks", []).append(toks.clone())
            trace_out.setdefault("raws", []).append(raws.clone())
        # --- 자식 삽입 (lane-major, c-minor — CPU append 순서 동일) ---
        lane_cell = f * W + torch.arange(W, device=dev)
        ar.cell.scatter_(0, sel.clamp(min=0),
                         torch.where(sel_valid, lane_cell,
                                     ar.cell.gather(0, sel.clamp(min=0))))
        ar.state.scatter_(0, sel.clamp(min=0),
                          torch.where(sel_valid, torch.ones_like(sel),
                                      ar.state.gather(0, sel.clamp(min=0))))
        offs = ar.n + torch.cumsum(fan, 0) - fan            # [W] excl.
        cgrid = torch.arange(c_tensor, device=dev)
        slot = offs.unsqueeze(1) + cgrid.unsqueeze(0)       # [W, C]
        child_ok = cgrid.unsqueeze(0) < fan.unsqueeze(1)    # [W, C]
        # boolean indexing 금지 (리뷰5: nonzero→DtoH 동기화 24회/rollout)
        # — 전 W×C 슬롯 밀집 쓰기, fan 밖 자식은 말단 scratch 슬롯으로.
        scratch = ar.capacity - 1
        sl = torch.where(child_ok, slot,
                         torch.full_like(slot, scratch)).reshape(-1)
        par = sel.unsqueeze(1).expand(W, c_tensor).reshape(-1)
        cix = cgrid.unsqueeze(0).expand(W, c_tensor).reshape(-1)
        tk = toks.reshape(-1)
        raws64 = raws.double()
        rq = raws64.reshape(-1)
        ok_q = (raws64 > 0.0).reshape(-1) & child_ok.reshape(-1)   # #38
        ar.tok.scatter_(0, sl, tk)
        ar.parent_idx.scatter_(0, sl, par)
        ar.depth.scatter_(0, sl, ar.depth.gather(0, par) + 1)
        ar.root.scatter_(0, sl, ar.root.gather(0, par))
        ar.sib.scatter_(0, sl, cix)
        child_lp = ar.logpri.gather(0, par) \
            + rq.clamp_min(1e-9).log()
        ar.logpri.scatter_(0, sl, torch.where(
            ok_q, child_lp, torch.full_like(child_lp, float("-inf"))))
        ar.raw_q.scatter_(0, sl, rq)
        ar.valid.scatter_(0, sl, ok_q)
        # 무효(zero-q·scratch) 슬롯은 평가 불가로 마킹 (선택 배제)
        ar.state.scatter_(0, sl, torch.where(
            ok_q, torch.zeros_like(sl), torch.ones_like(sl)))
        parent_anc = ar.anc_bits.index_select(0, par)
        parent_cell = ar.cell.gather(0, par).clamp(min=0)
        cell_word = torch.div(
            parent_cell, _ANC_WORD_BITS, rounding_mode="floor")
        cell_bit = parent_cell.remainder(_ANC_WORD_BITS)
        child_anc = parent_anc.clone()
        row = torch.arange(par.numel(), device=dev)
        child_anc[row, cell_word] |= torch.ones_like(cell_bit) << cell_bit
        ar.anc_bits.index_copy_(0, sl, child_anc)
        ar.n = ar.n + fan.sum()
        # backbone tip 전진: tip lane의 맏이(c=0). 주의 — pad lane의
        # r_of가 0으로 라우팅되므로 scatter_(중복 승자 미정)는 root0의
        # tip을 낡은 값으로 덮을 수 있다. tip은 root당 최대 1 lane
        # 이므로 델타 scatter_add(나머지는 +0)로 중복-안전하게 갱신.
        if not global_round:
            tip_adv = sel_valid & (sel == tip_idx.gather(0, r_of)) \
                & (fan > 0)
            old_tip = tip_idx.gather(0, r_of)
            delta = torch.where(tip_adv, offs - old_tip,
                                torch.zeros_like(offs))
            tip_idx.scatter_add_(0, r_of, delta)
            tip_depth.scatter_add_(0, r_of, tip_adv.long())
        sel_tr.append(sel)
        val_tr.append(sel_valid)
        fan_tr.append(fan)
    trace = (torch.stack(sel_tr), torch.stack(val_tr),
             torch.stack(fan_tr))
    return ar, trace, cell_logits
