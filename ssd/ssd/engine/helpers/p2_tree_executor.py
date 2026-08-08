"""23번 단계 2 — 전체-P2 CUDA graph 실행기 (실모델, 단일 버킷 v1).

캡처 범위: [arena reset → GPU 예산 → (select → fanout → 입력/rope
gather → packed mask 기록 → raw draft forward(KV 기록→attention) →
compute_logits → WOR 샘플(전용 generator) → 자식 삽입 + [R,Nv] 직접
기록) ×F].

계약 (검증된 전제 위에서):
- plan은 capture 시 버킷당 1회 (runtime plan 0회) — page-ID/slot/
  mask/입력은 고정 주소 버퍼, 내용은 replay 전 host copy로 갱신.
- wrapper: backend="fa2" (auto는 +64MB/wrapper), use_cuda_graph=True.
- RNG: 전용 generator + register_generator_state (기본 RNG 무오염
  참증명 통과 패턴).
- set_context는 라운드당 capture 시 1회 (replay 중 파이썬 0).
- 미지원 조건(비-Llama draft/EAGLE/B>1/temp0/페이지 초과)은 호출측
  arena fallback.

최종 범위: [W,Nv] tok/par/sib/raw_q/parent_cell/valid, parent-q
uniq(U-slot) 매핑, chain 호환 backbone token/logits까지 graph가 직접
기록한다. 실제 root는 앞 R행에만 존재하고 W-R행은 항상 무효인
padding이다. 이 물리 폭 W 계약은 cache key/layout 폭과 같아야 한다.
호출측은 고정 출력 버퍼의 view만 넘기며 CPU 변환이 없다.
"""
import os
import torch
import triton
import triton.language as tl

import flashinfer

from ssd.utils.context import set_context, reset_context
from ssd.engine.helpers import p2_tree as PT


@triton.jit
def _reset_tree_executor_kernel(
    parent_idx_ptr, parent_cell_ptr, logpri_ptr, state_ptr, cell_ptr,
    valid_ptr, anc_ptr, n_ptr, local_idx_ptr,
    out_tok_ptr, out_par_ptr, out_sib_ptr, out_rawq_ptr, out_pcell_ptr,
    out_valid_ptr,
    ARENA_N: tl.constexpr, ANC_N: tl.constexpr,
    OUT_N: tl.constexpr, OUT_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse the fixed-address arena/output reset into one launch.

    A CUDA graph removes host launch overhead, but the GPU still executes
    every captured ``fill_``/``zero_`` as a separate tiny kernel.  These
    fields have independent, constant reset values and are safe to clear in
    one bandwidth-light kernel before each four-round rollout.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    am = i < ARENA_N
    tl.store(parent_idx_ptr + i, -1, mask=am)
    tl.store(parent_cell_ptr + i, -1, mask=am)
    tl.store(logpri_ptr + i, float("-inf"), mask=am)
    tl.store(state_ptr + i, 0, mask=am)
    tl.store(cell_ptr + i, -1, mask=am)
    tl.store(valid_ptr + i, 0, mask=am)
    tl.store(anc_ptr + i, 0, mask=i < ANC_N)
    tl.store(local_idx_ptr + i, -1, mask=am)

    om = i < OUT_N
    tl.store(out_tok_ptr + i, 0, mask=om)
    tl.store(out_par_ptr + i, -1, mask=om)
    tl.store(out_sib_ptr + i, 0, mask=om)
    tl.store(out_rawq_ptr + i, 0.0, mask=om)
    tl.store(out_pcell_ptr + i, -1, mask=om)
    tl.store(out_valid_ptr + i, 0, mask=i < OUT_ROWS)
    tl.store(n_ptr, 0, mask=tl.program_id(0) == 0)


@triton.jit
def _insert_tree_children_kernel(
    # round inputs
    sel_ptr, sel_valid_ptr, fan_ptr, toks_ptr, rawq_ptr, child_lp_ptr,
    # arena state
    tok_ptr, parent_idx_ptr, depth_ptr, root_ptr, sib_ptr, logpri_ptr,
    arena_rawq_ptr, state_ptr, cell_ptr, valid_ptr, anc_ptr, n_ptr,
    local_idx_ptr,
    # fixed output views
    out_tok_ptr, out_par_ptr, out_sib_ptr, out_rawq_ptr, out_pcell_ptr,
    out_valid_ptr,
    # optional backbone-tip state used by non-global policies
    tip_idx_ptr, tip_depth_ptr,
    ROUND: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
    C: tl.constexpr, NV: tl.constexpr, ANC_WORDS: tl.constexpr,
    GLOBAL: tl.constexpr,
):
    """Insert one sampled round and record root-local views in one launch.

    The old fixed-shape PyTorch expression expanded this tiny W*C operation
    into scatter/gather/one_hot/cumsum kernels.  One scalar Triton program is
    intentional here: W<=10 and C<=3 in the supported executor, so serial
    bookkeeping is far cheaper than launching and synchronising dozens of
    kernels.  Model forward and sampling stay unchanged.
    """
    n0 = tl.load(n_ptr)
    total_fan = tl.full((), 0, tl.int64)

    # Mark selected parents as expanded and assign their forward-cell ids.
    for lane in tl.static_range(0, W):
        sv = tl.load(sel_valid_ptr + lane)
        parent = tl.load(sel_ptr + lane)
        tl.store(cell_ptr + parent, ROUND * W + lane, mask=sv)
        tl.store(state_ptr + parent, 1, mask=sv)

    # Insert children in the exact historical lane-major, then sibling-major
    # order.  Root-local rank is counted from the pre-round out_valid value;
    # out_valid itself is updated only after all children, avoiding atomics.
    for lane in tl.static_range(0, W):
        sv = tl.load(sel_valid_ptr + lane)
        parent = tl.load(sel_ptr + lane)
        nf = tl.load(fan_ptr + lane)
        root = tl.load(root_ptr + parent, mask=sv, other=0)
        pdepth = tl.load(depth_ptr + parent, mask=sv, other=0)
        pcell = tl.load(cell_ptr + parent, mask=sv, other=-1)
        parent_local = tl.load(local_idx_ptr + parent, mask=sv, other=-1)

        for child in tl.static_range(0, C):
            flat = lane * C + child
            active = sv & (child < nf)
            slot = n0 + total_fan + child
            rq = tl.load(rawq_ptr + flat)
            ok = active & (rq > 0.0)

            tl.store(tok_ptr + slot, tl.load(toks_ptr + flat), mask=active)
            tl.store(parent_idx_ptr + slot, parent, mask=active)
            tl.store(depth_ptr + slot, pdepth + 1, mask=active)
            tl.store(root_ptr + slot, root, mask=active)
            tl.store(sib_ptr + slot, child, mask=active)
            tl.store(logpri_ptr + slot, tl.load(child_lp_ptr + flat),
                     mask=active)
            tl.store(arena_rawq_ptr + slot, rq, mask=active)
            tl.store(valid_ptr + slot, ok, mask=active)
            tl.store(state_ptr + slot, tl.where(ok, 0, 1), mask=active)
            safe_pcell = tl.maximum(pcell, 0)
            pword = safe_pcell // 63
            pbit = safe_pcell - pword * 63
            for aw in tl.static_range(0, ANC_WORDS):
                panc = tl.load(
                    anc_ptr + parent * ANC_WORDS + aw,
                    mask=sv, other=0)
                add = tl.where(aw == pword, 1 << pbit, 0)
                tl.store(
                    anc_ptr + slot * ANC_WORDS + aw, panc | add,
                    mask=active)

            # Number of accepted children of the same root preceding this
            # item in lane-major order.  Static bounds make this a tiny
            # compile-time-unrolled scan and preserve exact view ordering.
            prior = tl.full((), 0, tl.int64)
            for prev_lane in tl.static_range(0, W):
                psv = tl.load(sel_valid_ptr + prev_lane)
                pp = tl.load(sel_ptr + prev_lane)
                pnf = tl.load(fan_ptr + prev_lane)
                proot = tl.load(root_ptr + pp, mask=psv, other=-1)
                for prev_child in tl.static_range(0, C):
                    precedes = ((prev_lane < lane) | ((prev_lane == lane)
                                & (prev_child < child)))
                    prq = tl.load(rawq_ptr + prev_lane * C + prev_child)
                    prior += (precedes & psv & (prev_child < pnf)
                              & (prq > 0.0) & (proot == root)).to(tl.int64)

            local = tl.load(out_valid_ptr + root, mask=ok, other=0) + prior
            write = ok & (local < NV)
            dst = root * NV + local
            tl.store(out_tok_ptr + dst, tl.load(toks_ptr + flat), mask=write)
            tl.store(out_par_ptr + dst, parent_local, mask=write)
            tl.store(out_sib_ptr + dst, child, mask=write)
            tl.store(out_rawq_ptr + dst, rq, mask=write)
            tl.store(out_pcell_ptr + dst, pcell, mask=write)
            # Every active slot receives a deterministic local marker, even
            # if an invalid/overflow child is not exposed in the view.
            tl.store(local_idx_ptr + slot, tl.where(write, local, -1),
                     mask=active)

        # The current lane's children occupy a contiguous arena range.
        if not GLOBAL:
            is_tip = sv & (parent == tl.load(tip_idx_ptr + root)) & (nf > 0)
            tl.store(tip_idx_ptr + root, n0 + total_fan, mask=is_tip)
            old_depth = tl.load(tip_depth_ptr + root, mask=is_tip, other=0)
            tl.store(tip_depth_ptr + root, old_depth + 1, mask=is_tip)
        total_fan += nf

    # Commit accepted counts after every local rank has read the old base.
    for rr in tl.static_range(0, R):
        accepted = tl.full((), 0, tl.int64)
        for lane in tl.static_range(0, W):
            sv = tl.load(sel_valid_ptr + lane)
            pp = tl.load(sel_ptr + lane)
            nf = tl.load(fan_ptr + lane)
            proot = tl.load(root_ptr + pp, mask=sv, other=-1)
            for child in tl.static_range(0, C):
                rq = tl.load(rawq_ptr + lane * C + child)
                accepted += (sv & (child < nf) & (rq > 0.0)
                             & (proot == rr)).to(tl.int64)
        old = tl.load(out_valid_ptr + rr)
        tl.store(out_valid_ptr + rr, old + accepted)
    tl.store(n_ptr, n0 + total_fan)


@triton.jit
def _mark_selected_parents_kernel(
    sel_ptr, sel_valid_ptr, cell_ptr, state_ptr,
    ROUND: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr,
):
    """Scalable parent marking used by wide/multiword P1 shapes."""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = lane < W
    sv = tl.load(sel_valid_ptr + lane, mask=live, other=0)
    parent = tl.load(sel_ptr + lane, mask=live, other=0)
    tl.store(cell_ptr + parent, ROUND * W + lane, mask=live & sv)
    tl.store(state_ptr + parent, 1, mask=live & sv)


@triton.jit
def _insert_tree_children_parallel_kernel(
    sel_ptr, sel_valid_ptr, fan_ptr, toks_ptr, rawq_ptr, child_lp_ptr,
    tok_ptr, parent_idx_ptr, depth_ptr, root_ptr, sib_ptr, logpri_ptr,
    arena_rawq_ptr, state_ptr, cell_ptr, valid_ptr, anc_ptr, n_ptr,
    local_idx_ptr,
    out_tok_ptr, out_par_ptr, out_sib_ptr, out_rawq_ptr, out_pcell_ptr,
    out_valid_ptr,
    W: tl.constexpr, C: tl.constexpr, NV: tl.constexpr,
    ANC_WORDS: tl.constexpr,
):
    """One program per possible child for P1-scale fixed shapes.

    The P2 scalar kernel intentionally unrolls the complete W*C bookkeeping
    because W<=10.  Doing that for P1 W=20,F=9 made ptxas spend minutes on a
    huge program.  This kernel keeps identical lane-major ordering while
    parallelizing children; the following count kernel commits per-root sizes.
    """
    flat = tl.program_id(0)
    lane = flat // C
    child = flat - lane * C
    sv = tl.load(sel_valid_ptr + lane)
    nf = tl.load(fan_ptr + lane)
    active = sv & (child < nf)
    parent = tl.load(sel_ptr + lane)
    root = tl.load(root_ptr + parent, mask=sv, other=0)

    n0 = tl.load(n_ptr)
    prefix = tl.full((), 0, tl.int64)
    for prev_lane in tl.static_range(0, W):
        prefix += tl.where(prev_lane < lane,
                           tl.load(fan_ptr + prev_lane), 0)
    slot = n0 + prefix + child
    rq = tl.load(rawq_ptr + flat)
    ok = active & (rq > 0.0)
    pdepth = tl.load(depth_ptr + parent, mask=sv, other=0)
    pcell = tl.load(cell_ptr + parent, mask=sv, other=-1)
    parent_local = tl.load(local_idx_ptr + parent, mask=sv, other=-1)

    tl.store(tok_ptr + slot, tl.load(toks_ptr + flat), mask=active)
    tl.store(parent_idx_ptr + slot, parent, mask=active)
    tl.store(depth_ptr + slot, pdepth + 1, mask=active)
    tl.store(root_ptr + slot, root, mask=active)
    tl.store(sib_ptr + slot, child, mask=active)
    tl.store(logpri_ptr + slot, tl.load(child_lp_ptr + flat), mask=active)
    tl.store(arena_rawq_ptr + slot, rq, mask=active)
    tl.store(valid_ptr + slot, ok, mask=active)
    tl.store(state_ptr + slot, tl.where(ok, 0, 1), mask=active)

    safe_pcell = tl.maximum(pcell, 0)
    pword = safe_pcell // 63
    pbit = safe_pcell - pword * 63
    for aw in tl.static_range(0, ANC_WORDS):
        inherited = tl.load(
            anc_ptr + parent * ANC_WORDS + aw, mask=sv, other=0)
        add = tl.where(aw == pword, 1 << pbit, 0)
        tl.store(anc_ptr + slot * ANC_WORDS + aw, inherited | add,
                 mask=active)

    # Root-local rank is the old count plus valid children of this root that
    # precede ``flat`` in lane-major/sibling-major order.
    prior = tl.full((), 0, tl.int64)
    for prev in tl.static_range(0, W * C):
        pl = prev // C
        pc = prev - pl * C
        psv = tl.load(sel_valid_ptr + pl)
        pnf = tl.load(fan_ptr + pl)
        pp = tl.load(sel_ptr + pl)
        proot = tl.load(root_ptr + pp, mask=psv, other=-1)
        prq = tl.load(rawq_ptr + prev)
        prior += ((prev < flat) & psv & (pc < pnf) & (prq > 0.0)
                  & (proot == root)).to(tl.int64)
    local = tl.load(out_valid_ptr + root, mask=ok, other=0) + prior
    write = ok & (local < NV)
    dst = root * NV + local
    tl.store(out_tok_ptr + dst, tl.load(toks_ptr + flat), mask=write)
    tl.store(out_par_ptr + dst, parent_local, mask=write)
    tl.store(out_sib_ptr + dst, child, mask=write)
    tl.store(out_rawq_ptr + dst, rq, mask=write)
    tl.store(out_pcell_ptr + dst, pcell, mask=write)
    tl.store(local_idx_ptr + slot, tl.where(write, local, -1), mask=active)


@triton.jit
def _advance_backbone_tips_parallel_kernel(
    sel_ptr, sel_valid_ptr, fan_ptr, root_ptr, n_ptr,
    tip_idx_ptr, tip_depth_ptr,
    W: tl.constexpr,
):
    """Advance each root's mandatory first-child path after wide insertion.

    One root has at most one selected tip in a round, so these stores do not
    contend.  ``n_ptr`` still contains the pre-insertion arena length; the
    commit kernel runs afterwards.
    """
    lane = tl.program_id(0)
    sv = tl.load(sel_valid_ptr + lane)
    nf = tl.load(fan_ptr + lane)
    parent = tl.load(sel_ptr + lane)
    root = tl.load(root_ptr + parent, mask=sv, other=0)
    old_tip = tl.load(tip_idx_ptr + root, mask=sv, other=-1)
    is_tip = sv & (nf > 0) & (parent == old_tip)

    prefix = tl.full((), 0, tl.int64)
    for prev_lane in tl.static_range(0, W):
        prefix += tl.where(
            prev_lane < lane, tl.load(fan_ptr + prev_lane), 0)
    first_child = tl.load(n_ptr) + prefix
    tl.store(tip_idx_ptr + root, first_child, mask=is_tip)
    old_depth = tl.load(tip_depth_ptr + root, mask=is_tip, other=0)
    tl.store(tip_depth_ptr + root, old_depth + 1, mask=is_tip)


@triton.jit
def _commit_tree_children_parallel_kernel(
    sel_ptr, sel_valid_ptr, fan_ptr, rawq_ptr, root_ptr,
    out_valid_ptr, n_ptr,
    W: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
):
    """Commit per-root accepted counts and total arena slots after insertion."""
    rr = tl.program_id(0)
    accepted = tl.full((), 0, tl.int64)
    total_fan = tl.full((), 0, tl.int64)
    for lane in tl.static_range(0, W):
        sv = tl.load(sel_valid_ptr + lane)
        nf = tl.load(fan_ptr + lane)
        pp = tl.load(sel_ptr + lane)
        proot = tl.load(root_ptr + pp, mask=sv, other=-1)
        total_fan += nf
        for child in tl.static_range(0, C):
            rq = tl.load(rawq_ptr + lane * C + child)
            accepted += (sv & (child < nf) & (rq > 0.0)
                         & (proot == rr)).to(tl.int64)
    old = tl.load(out_valid_ptr + rr)
    tl.store(out_valid_ptr + rr, old + accepted)
    tl.store(n_ptr, tl.load(n_ptr) + total_fan, mask=rr == 0)


@triton.jit
def _finalize_tree_meta_kernel(
    tok_ptr, par_ptr, sib_ptr, pcell_ptr, valid_ptr,
    pq_ref_ptr, pq_cells_ptr, u_valid_ptr,
    backbone_tok_ptr, backbone_pcell_ptr,
    NV: tl.constexpr, F: tl.constexpr,
):
    """One program per root; NV/F are tiny compile-time constants."""
    r = tl.program_id(0)
    base = r * NV
    n = tl.load(valid_ptr + r)

    # Deterministic padding for the fixed-width wire buffers.
    for j in tl.static_range(0, NV):
        tl.store(pq_ref_ptr + base + j, -1)
        tl.store(pq_cells_ptr + base + j, -1)

    # First-occurrence parent-cell numbering (build_root_views contract).
    u = tl.full((), 0, tl.int64)
    for j in tl.static_range(0, NV):
        pc = tl.load(pcell_ptr + base + j)
        j_valid = j < n
        seen = tl.full((), 0, tl.int1)
        ref = u
        for k in tl.static_range(0, j):
            same = (k < n) & (tl.load(pcell_ptr + base + k) == pc)
            seen = seen | same
            ref = tl.where(same, tl.load(pq_ref_ptr + base + k), ref)
        is_new = j_valid & ~seen
        tl.store(pq_cells_ptr + base + u, pc, mask=is_new)
        tl.store(pq_ref_ptr + base + j,
                 tl.where(j_valid, ref, -1))
        u += is_new.to(tl.int64)
    tl.store(u_valid_ptr + r, u)

    # First-child backbone projection.  Save parent-cell ids; a second,
    # vocabulary-parallel kernel gathers the corresponding logits.
    cur = tl.full((), -1, tl.int64)
    for depth in tl.static_range(0, F):
        found = tl.full((), 0, tl.int1)
        child = tl.full((), -1, tl.int64)
        child_tok = tl.full((), 0, tl.int64)
        child_pc = tl.full((), -1, tl.int64)
        for j in tl.static_range(0, NV):
            take = ((j < n) & ~found
                    & (tl.load(par_ptr + base + j) == cur)
                    & (tl.load(sib_ptr + base + j) == 0))
            child = tl.where(take, j, child)
            child_tok = tl.where(take, tl.load(tok_ptr + base + j),
                                 child_tok)
            child_pc = tl.where(take, tl.load(pcell_ptr + base + j),
                                child_pc)
            found = found | take
        out = r * F + depth
        tl.store(backbone_tok_ptr + out, child_tok)
        tl.store(backbone_pcell_ptr + out, child_pc)
        cur = tl.where(found, child, -2)


@triton.jit
def _gather_backbone_logits_kernel(
    cell_logits_ptr, backbone_pcell_ptr, backbone_logits_ptr,
    SRC_ROWS: tl.constexpr, OUT_ROWS: tl.constexpr,
    V: tl.constexpr, BLOCK: tl.constexpr,
):
    """Gather selected vocabulary rows directly into the persistent output.

    There must be no full-vocabulary intermediate here.  P1 can expose more
    response nodes than its chain depth (for example 18 nodes at K1=9), so a
    PyTorch ``index_select`` followed by ``where`` temporarily materializes
    multiple [OUT_ROWS, V] float32 tensors and can exhaust a 24-GiB draft
    GPU after CUDA-graph warmup.

    The source row is checked on both sides *before* the load.  Invalid and
    padded backbone rows are written as zero.  A two-dimensional launch also
    keeps row/column arithmetic independent, avoiding the unchecked flattened
    ``pc * V + col`` implementation that was previously unsafe in graph use.
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_ok = col < V
    pc = tl.load(backbone_pcell_ptr + row, mask=row < OUT_ROWS, other=-1)
    src_ok = (pc >= 0) & (pc < SRC_ROWS)
    value = tl.load(
        cell_logits_ptr + pc * V + col,
        mask=col_ok & src_ok,
        other=0.0,
    )
    tl.store(
        backbone_logits_ptr + row * V + col,
        value,
        mask=(row < OUT_ROWS) & col_ok,
    )


def _gather_backbone_logits(cell_logits, backbone_pcell,
                            backbone_logits):
    """Launch the bounded, allocation-free backbone-logit gather."""
    src_rows, vocab = cell_logits.shape
    out_rows = backbone_pcell.numel()
    if backbone_logits.numel() != out_rows * vocab:
        raise RuntimeError(
            "backbone logit gather shape mismatch: "
            f"pcell={tuple(backbone_pcell.shape)} "
            f"out={tuple(backbone_logits.shape)} vocab={vocab}")
    block = 1024
    _gather_backbone_logits_kernel[
        (out_rows, triton.cdiv(vocab, block))
    ](
        cell_logits,
        backbone_pcell,
        backbone_logits,
        SRC_ROWS=src_rows,
        OUT_ROWS=out_rows,
        V=vocab,
        BLOCK=block,
        num_warps=8,
    )


class P2TreeExecutor:
    def __init__(self, model, compute_logits_fn, config, device,
                 block_size, max_blocks, vocab_size,
                 num_heads, num_kv_heads, head_dim,
                 dtype=torch.float16, *, phase="p2", width=None,
                 root_count=None, depth=None, max_nodes=None,
                 glue_width=None):
        self.model = model
        self.compute_logits = compute_logits_fn
        self.cfg = config
        self.dev = device
        self.bs = block_size
        self.max_blocks = max_blocks
        self.V = vocab_size
        self.H, self.HKV, self.D = num_heads, num_kv_heads, head_dim
        self.dtype = dtype
        if phase not in ("p1", "p2"):
            raise ValueError(f"dynamic tree phase must be p1|p2; got {phase}")
        self.phase = phase
        self.W = int(width if width is not None
                     else config.duet_proxy_total_budget)
        # Config owns the canonical active-root rule.  In particular,
        # confidence mode uses the same automatic R for the selector, eager
        # reference path, and captured executor; duplicating the old
        # ``confidence => W`` special case here silently changed topology.
        self.R = int(root_count if root_count is not None
                     else config.duet_p2_seed_count)
        self.F = int(depth if depth is not None else config.duet_phase2_k)
        self.C = int(config.duet_tree_c_tensor)
        self.NV = int(
            max_nodes if max_nodes is not None
            else (getattr(config, "duet_p1_tree_max_nodes",
                          getattr(config, "duet_tree_nv", 8))
                  if phase == "p1" else
                  getattr(config, "duet_p2_tree_max_nodes",
                          getattr(config, "duet_tree_nv", 8))))
        # Both phases use the same global dynamic expansion.  P1 supplies a
        # draft-derived context-reach/root prior where P2 supplies the target
        # proxy prior.  Legacy selectors remain available only for controlled
        # reproduction tests.
        self.policy = ("dynamic" if phase == "p1" else
                       getattr(config, "duet_tree_policy", "eagle"))
        W, R, F, C, NV = self.W, self.R, self.F, self.C, self.NV
        if R > W:
            raise ValueError(
                f"{phase} dynamic tree requires roots R<=forward width W; "
                f"got R={R}, W={W}")
        d = device
        # ── 입력 고정 버퍼 (replay 전 host가 내용 갱신)
        self.in_root_tok = torch.zeros(R, dtype=torch.int64, device=d)
        self.in_root_piv = torch.zeros(R, dtype=torch.float32, device=d)
        self.in_rope_base = torch.zeros(R, dtype=torch.int64, device=d)
        gw_max = int(glue_width) if glue_width is not None else (
            max(int(getattr(config, "duet_phase1_k", None) or F),
                int(getattr(config, "duet_phase2_k", F)),
                int(getattr(config, "duet_p1_tree_max_nodes", F)),
                int(getattr(config, "duet_p2_tree_max_nodes", F))) + 1)
        self.gw_max = gw_max
        self.in_glue = torch.zeros(R, gw_max, dtype=torch.uint8,
                                   device=d)
        # Persistent pinned staging avoids constructing a short-lived CUDA
        # tensor from the NumPy glue matrix on every P2 step.
        self.in_glue_host = torch.empty(R, gw_max, dtype=torch.uint8,
                                        pin_memory=True)
        # 실 glue 폭 (요청별 상이 — spec 열 정렬에 필요; 내용-구동)
        self.in_glue_w = torch.zeros(1, dtype=torch.int64, device=d)
        self.in_temps = torch.zeros(W, dtype=torch.float32, device=d)
        self.in_slot = [torch.zeros(W, dtype=torch.int32, device=d)
                        for _ in range(F)]
        self.in_ctx_len = [torch.zeros(1, dtype=torch.int32, device=d)
                           for _ in range(F)]
        self.in_block_tables = torch.zeros(1, max_blocks,
                                           dtype=torch.int32, device=d)
        self.in_page_ids = torch.empty(max_blocks, dtype=torch.int32,
                                       device=d)
        # prefix 경계는 버킷 내 요청마다 달라짐 — 캡처에 박히면 안
        # 되는 '내용' (버퍼 구동; python int 슬라이싱 금지)
        self.in_prefix_len = torch.zeros(1, dtype=torch.int64, device=d)
        # ── arena / RNG / 출력
        self.arena = PT.TreeArena(
            R + F * W * C, d, max_cells=F * W)
        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(torch.initial_seed() % (2**31))
        # 출력의 물리 행 수는 root 수 R이 아니라 layout/cache-key 폭 W.
        # R<W일 때도 소비자는 proxy root id 0..W-1로 뷰를 조회한다.
        # 앞 R행만 실제 root이고 뒤 W-R행은 valid=0 padding이다.
        out_r = W
        self.out_rows = out_r
        # +1 더미 슬롯: 제외 항목 라우팅용 (index-0 충돌 방지 —
        # 중복-scatter 승자미정, tip 버그와 동일 계열; 판별 parity로
        # 발견). view는 [:, :NV]만 소비.
        self.out_tok = torch.zeros(out_r * NV + 1, dtype=torch.int64,
                                   device=d)
        self.out_par = torch.full((out_r * NV + 1,), -1,
                                  dtype=torch.int64,
                                  device=d)
        self.out_sib = torch.zeros(out_r * NV + 1, dtype=torch.int64,
                                   device=d)
        self.out_rawq = torch.zeros(out_r * NV + 1, dtype=torch.float32,
                                    device=d)
        self.out_pcell = torch.full((out_r * NV + 1,), -1,
                                    dtype=torch.int64,
                                    device=d)
        self.out_valid = torch.zeros(out_r, dtype=torch.int64, device=d)

        # 소비자용 [W,NV] 뷰 (더미 슬롯 제외)
        self.view_tok = self.out_tok[:out_r * NV].view(out_r, NV)
        self.view_par = self.out_par[:out_r * NV].view(out_r, NV)
        self.view_sib = self.out_sib[:out_r * NV].view(out_r, NV)
        self.view_rawq = self.out_rawq[:out_r * NV].view(out_r, NV)
        self.view_pcell = self.out_pcell[:out_r * NV].view(out_r, NV)
        self.cell_logits = torch.zeros(F * W, vocab_size,
                                       dtype=torch.float32, device=d)
        # 최종 소비자 계약도 graph 안에서 직접 생성한다.  이 버퍼들은
        # replay마다 같은 주소에서 갱신되며, 호출측은 참조만 전달한다.
        self.out_pq_ref = torch.full((out_r, NV), -1, dtype=torch.int64,
                                     device=d)
        self.out_pq_cells = torch.full((out_r, NV), -1,
                                       dtype=torch.int64,
                                       device=d)
        self.out_u_valid = torch.zeros(out_r, dtype=torch.int64, device=d)
        self.out_backbone_tok = torch.zeros(out_r, F, dtype=torch.int64,
                                            device=d)
        self.out_backbone_pcell = torch.full(
            (out_r, F), -1, dtype=torch.int64, device=d)
        self.out_backbone_logits = torch.zeros(
            out_r, F, vocab_size, dtype=dtype, device=d)
        # 단계1 진단 버퍼 (in-graph 기록 — replay마다 갱신, 비용 미미)
        F_, W_, C_ = self.F, self.W, self.C
        self.dbg_ids = torch.zeros(F_, W_, dtype=torch.int64, device=d)
        self.dbg_rope = torch.zeros(F_, W_, dtype=torch.int64, device=d)
        self.dbg_toks = torch.zeros(F_, W_, C_, dtype=torch.int64,
                                    device=d)
        self.dbg_raws = torch.zeros(F_, W_, C_, dtype=torch.float32,
                                    device=d)
        self.dbg_fan = torch.zeros(F_, W_, dtype=torch.int64, device=d)
        self.dbg_sel = torch.zeros(F_, W_, dtype=torch.int64, device=d)
        self.dbg_selv = torch.zeros(F_, W_, dtype=torch.bool, device=d)

        # Arena node -> root-local output index.  CUDA graphs for different
        # page buckets must not share mutable internal state.  Keep one live
        # tensor per bucket and point ``_local_idx`` at the bucket being
        # captured/replayed so diagnostics observe the right graph state.
        # (A prior single shared tensor made seven pre-captured real-model
        # graphs intermittently illegal-access; the original per-capture
        # tensors were safe but became unreachable to the audit.)
        self._local_idx = torch.full((self.arena.capacity,), -1,
                                     dtype=torch.int64, device=d)
        self._local_idx_by_bucket = {}

        # 캡처 호환 상수
        self.ones_w = torch.ones(W, dtype=torch.uint8, device=d)
        self.lane_w = torch.arange(W, device=d)
        self.arange_R = torch.arange(R, device=d)
        self.graphs = {}                       # n_pages → CUDAGraph
        # 결정적 parity 모드 (리뷰12 §3): round별 [W,V] 고정 noise
        # 버퍼 — None이면 프로덕션 RNG. 주소는 캡처에 박히고 내용은
        # 교체 가능 → eager/replay 동일 noise 주입 비교.
        self.parity_noise = None
        self.wrappers = {}                     # n_pages → [wrapper]*F
        # FlashInfer wrappers keep plan-specific auxiliary state.  Sharing a
        # single workspace across wrappers of the *same* page shape is safe
        # because all rounds use the same plan and execute serially.  Sharing
        # it across different page shapes leaves graph lifetime dependent on
        # FlashInfer's workspace reuse details.  That was one of two mutable
        # aliases in the failing all-bucket path.  Isolate it so every graph
        # remains self-contained regardless of backend implementation.
        #
        # Keep one modest FA2 workspace per page bucket.  The production
        # TinyLlama shape requests about 41 MiB for batch_prefill_tmp_v (the
        # mini-model tests request much less), so retain a 64 MiB safety
        # margin while avoiding 7*128 MiB.  The size remains overridable for
        # future model shapes.
        self._workspace_bytes = int(os.environ.get(
            "SSD_TREE_EXEC_WORKSPACE_MB", "64")) * 2**20
        if self._workspace_bytes <= 0:
            raise ValueError("SSD_TREE_EXEC_WORKSPACE_MB must be positive")
        self._float_ws_by_bucket = {}

    # ---------- 버킷 준비 ----------
    def _mk_round_wrapper(self, n_pages_r, canvas_cols, float_workspace):
        d = self.dev
        W = self.W
        qo = torch.tensor([0, W], dtype=torch.int32, device=d)
        kvp = torch.tensor([0, n_pages_r], dtype=torch.int32, device=d)
        kvi = torch.zeros(n_pages_r, dtype=torch.int32, device=d)
        lpl = torch.tensor([self.bs], dtype=torch.int32, device=d)
        n_pk = (W * canvas_cols + 7) // 8
        mask_buf = torch.zeros(n_pk, dtype=torch.uint8, device=d)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace, "NHD", backend="fa2", use_cuda_graph=True,
            qo_indptr_buf=qo, paged_kv_indptr_buf=kvp,
            paged_kv_indices_buf=kvi, paged_kv_last_page_len_buf=lpl,
            custom_mask_buf=mask_buf,
            mask_indptr_buf=torch.tensor([0, W * canvas_cols],
                                         dtype=torch.int32, device=d))
        wr.plan(qo, kvp, kvi, lpl,
                self.H, self.HKV, self.D, self.bs,
                custom_mask=torch.zeros(W * canvas_cols,
                                        dtype=torch.bool, device=d),
                q_data_type=self.dtype, kv_data_type=self.dtype)
        wr._canvas_cols = canvas_cols
        return wr

    def prepare_bucket(self, n_pages0):
        """버킷 = 시작 page 수. round r의 canvas = (p0+1) page 고정
        (전제 검증: p+1 전체-page 0-mask 안전)."""
        F, W = self.F, self.W
        float_workspace = self._float_ws_by_bucket.get(n_pages0)
        if float_workspace is None:
            float_workspace = torch.empty(
                self._workspace_bytes, dtype=torch.uint8, device=self.dev)
            self._float_ws_by_bucket[n_pages0] = float_workspace
        wrappers = []
        for f in range(F):
            canvas_pages = n_pages0 + 1
            wrappers.append(self._mk_round_wrapper(
                canvas_pages, canvas_pages * self.bs, float_workspace))
        self.wrappers[n_pages0] = wrappers
        return wrappers

    @torch.inference_mode()
    def prime_capture_inputs(self, n_pages0):
        """Install a finite, fully-active synthetic request before capture.

        Startup capture previously used constructor-zero state: context
        length zero, repeated slot zero and no productive tree nodes.  That
        is not a valid representative of the kernels replayed in service and
        made all-bucket capture depend on data-degenerate launch behaviour;
        use production-valid launch geometry instead.
        KV page 0 is zero-initialized by ModelRunner, so mapping every canvas
        page to it is finite and safe; the custom mask exposes only the
        synthetic cells written below.
        """
        if n_pages0 not in self.wrappers:
            self.prepare_bucket(n_pages0)
        self.in_root_tok.zero_()
        self.in_root_piv.fill_(1.0 / max(1, self.R))
        self.in_rope_base.zero_()
        self.in_glue.zero_()
        self.in_glue_w.zero_()
        self.in_temps.fill_(0.7)
        self.in_prefix_len.zero_()
        self.in_block_tables.zero_()
        for f in range(self.F):
            self.in_slot[f].copy_(f * self.W + self.lane_w)
            self.in_ctx_len[f].fill_((f + 1) * self.W)
            self.wrappers[n_pages0][f]._paged_kv_indices_buf.zero_()

    # ---------- 본체 (캡처 대상) ----------
    def _pack_row_mask(self, wr, f):
        """행별 [prefix 1s | glue | 조상셀 | self] canvas — 열 배치가
        prefix 길이(요청별 상이)에 의존하므로 전부 버퍼-구동 텐서
        연산 (python int 슬라이싱은 캡처에 박힘 — 금지)."""
        W, F = self.W, self.F
        canvas = wr._canvas_cols
        ar = self.arena
        sel, sel_valid = self._sel[f]
        plen = self.in_prefix_len                     # [1] int64 버퍼
        col = torch.arange(canvas, device=self.dev)   # [canvas]
        gW_max = self.in_glue.shape[1]
        gW = self.in_glue_w                            # [1] 실폭 (버퍼)
        r_of = torch.where(sel_valid,
                           ar.root.gather(0, sel.clamp(min=0)),
                           torch.zeros_like(sel))
        # prefix: col < plen
        m = (col.unsqueeze(0) < plen).expand(W, canvas) \
            .to(torch.uint8).clone()
        # glue: col ∈ [plen, plen+gW실폭) → glue[r_of, col-plen]
        g_off = col.unsqueeze(0) - plen               # [1, canvas]
        in_glue_rng = (g_off >= 0) & (g_off < gW)
        g_idx = g_off.clamp(min=0, max=gW_max - 1)
        g_bits = self.in_glue.index_select(0, r_of) \
            .gather(1, g_idx.expand(W, canvas)) \
            * sel_valid.unsqueeze(1).to(torch.uint8)
        m = torch.where(in_glue_rng.expand(W, canvas), g_bits, m)
        # 조상 셀: col ∈ [plen+gW, plen+gW+f·W) — 실폭 기준 정렬
        spec_off = g_off - gW
        anc = ar.anc_bits.index_select(0, sel.clamp(min=0)) \
            * sel_valid.long().unsqueeze(1)
        in_spec = (spec_off >= 0) & (spec_off < f * W) if f else \
            torch.zeros(1, canvas, dtype=torch.bool, device=self.dev)
        if f:
            safe_off = spec_off.clamp(min=0, max=max(f * W - 1, 0))
            a_word = torch.div(
                safe_off, PT._ANC_WORD_BITS, rounding_mode="floor")
            a_bit = safe_off.remainder(PT._ANC_WORD_BITS)
            anc_word = anc.gather(1, a_word.expand(W, canvas))
            a_bits = ((anc_word >> a_bit.expand(W, canvas)) & 1) \
                .to(torch.uint8)
            m = torch.where(in_spec.expand(W, canvas), a_bits, m)
        # self 셀: col == plen+gW(실폭)+f·W+lane — 비활성 lane은 0
        # (arena _arena_mask_pack의 `(bits|selfbit)*sel_valid` 규약;
        # 단계0 강화 mask 대조가 잡은 최초 불일치 — 비활성 lane
        # self=1은 valid-lane logits에는 무영향이나 mask bytes exact
        # 계약 위반)
        self_col = plen + gW + f * W + self.lane_w.unsqueeze(1)  # [W,1]
        is_self = col.unsqueeze(0) == self_col        # [W, canvas]
        m = torch.where(is_self,
                        sel_valid.to(torch.uint8).unsqueeze(1)
                        .expand(W, canvas), m)
        flat = m.reshape(-1)
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, torch.zeros(
                pad, dtype=torch.uint8, device=self.dev)])
        wbits = (1 << torch.arange(8, device=self.dev)).to(torch.uint8)
        packed = (flat.view(-1, 8) * wbits).sum(
            1, dtype=torch.int64).to(torch.uint8)
        wr._custom_mask_buf[:packed.numel()].copy_(packed)

    def run_once(self, n_pages0, finalize=True):
        """Execute the fixed-shape P2 body once.

        ``finalize=False`` is used for CUDA graph capture.  The four draft
        rounds and all inter-round dependencies remain captured, while the
        final data-dependent row gather is launched immediately after replay.
        Capturing that last gather caused intermittent graph-only illegal
        accesses even though the same indices were valid in eager execution.
        """
        W, R, F, C, NV = self.W, self.R, self.F, self.C, self.NV
        ar = self.arena
        _reset_n = max(ar.capacity, ar.anc_bits.numel(),
                       self.out_tok.numel(),
                       self.out_valid.numel())
        _reset_tree_executor_kernel[(triton.cdiv(_reset_n, 256),)](
            ar.parent_idx, ar.parent_cell, ar.logpri, ar.state, ar.cell,
            ar.valid, ar.anc_bits, ar.n, self._local_idx,
            self.out_tok, self.out_par, self.out_sib, self.out_rawq,
            self.out_pcell, self.out_valid,
            ARENA_N=ar.capacity, ANC_N=ar.anc_bits.numel(),
            OUT_N=self.out_tok.numel(),
            OUT_ROWS=self.out_valid.numel(), BLOCK=256)
        policy = self.policy
        if policy in ("coverage", "backbone", "dynamic", "eagle", "hybrid",
                      "adaptive"):
            # Stored children are not forward cells.  A single parent
            # forward already samples C ordered WOR children, so retaining
            # siblings up to NV does not add model calls or change the fixed
            # F x W CUDA graph shape.
            budgets = torch.where(
                self.in_root_piv > 0,
                torch.full_like(self.in_root_piv, NV, dtype=torch.int64),
                torch.zeros_like(self.in_root_piv, dtype=torch.int64))
        else:
            _beta = (0.5 if policy == "confidence"
                     else self.cfg.duet_tree_beta)
            budgets = PT.alloc_root_budgets_gpu(
                self.in_root_piv, total=F * W, beta=_beta, cap=NV)
        remaining = budgets.clone()
        ar.tok[:R] = self.in_root_tok
        ar.root[:R] = self.arange_R
        ar.logpri[:R] = self.in_root_piv.clamp_min(1e-9) \
            .log().double()
        # Keep physical roots present for arena/reference index parity.  A
        # zero P_iv gives them zero child budget, while sanitized token/rope
        # values keep their fixed-width padding forwards model-safe.
        ar.valid[:R] = True
        ar.n += R
        tip_idx = self.arange_R.clone()
        tip_depth = torch.zeros(R, dtype=torch.int64, device=self.dev)
        self._sel = {}
        wrappers = self.wrappers[n_pages0]
        hybrid_floor = min(2, F)
        for f in range(F):
            _global = (policy in ("dynamic", "eagle")
                       or (policy == "hybrid" and f >= hybrid_floor))
            if _global:
                sel, sel_valid = PT._arena_select_global(
                    ar, W, f, F, remaining,
                    future_rounds=F - f - 1, R=R,
                    proxy_threshold=(
                        0.0 if self.phase == "p1" else float(getattr(
                            self.cfg, "duet_tree_proxy_threshold", 0.0))),
                    conf_threshold=float(getattr(
                        self.cfg, "duet_tree_conf_threshold", 0.0)))
            else:
                sel, sel_valid = PT._arena_select(
                    ar, "level", W, f, F, tip_idx, remaining)
            self._sel[f] = (sel, sel_valid)
            if _global:
                fan = PT._arena_fanout_global(
                    ar, sel, sel_valid, remaining, C, R,
                    future_rounds=F - f - 1)
            else:
                if policy == "hybrid":
                    r_sel = ar.root.gather(0, sel.clamp(min=0))
                    is_tip = sel_valid & (sel == tip_idx.gather(0, r_sel))
                    fan = (is_tip & (remaining.gather(0, r_sel) > 0)).long()
                else:
                    reserve = (F - tip_depth).clamp(min=0)
                    _fanout_fn = (PT._arena_fanout_adaptive
                                  if policy == "adaptive"
                                  else PT._arena_fanout_backbone)
                    fan = _fanout_fn(
                        ar, sel, sel_valid, tip_idx, remaining, reserve, C, R)
            r_of = torch.where(sel_valid,
                               ar.root.gather(0, sel.clamp(min=0)),
                               torch.zeros_like(sel))
            remaining.scatter_add_(0, r_of, -fan)
            ids = torch.where(sel_valid,
                              ar.tok.gather(0, sel.clamp(min=0)),
                              torch.zeros_like(sel))
            rope = torch.where(
                sel_valid,
                self.in_rope_base.gather(0, r_of)
                + ar.depth.gather(0, sel.clamp(min=0)),
                self.in_rope_base[0].expand(W))
            self.dbg_sel[f].copy_(sel)
            self.dbg_selv[f].copy_(sel_valid)
            self.dbg_ids[f].copy_(ids)
            self.dbg_rope[f].copy_(rope)
            self.dbg_fan[f].copy_(fan)
            self._pack_row_mask(wrappers[f], f)
            # ── raw draft forward (capture-시 context 1회 bake)
            set_context(
                is_prefill=False,
                slot_mapping=self.in_slot[f],
                context_lens=self.in_ctx_len[f],
                block_tables=self.in_block_tables,
                active_mq_len=W,
                active_wrappers={1: wrappers[f]},
            )
            try:
                hidden = self.model(ids, rope)
                logits = self.compute_logits(hidden, False)[:W].float()
            finally:
                reset_context()
            self.cell_logits[f * W:(f + 1) * W] = logits
            toks, raws = PT.tree_sample_wor(
                logits, self.in_temps, C, assume_pos_temps=True,
                sampler_x=getattr(self.cfg, "sampler_x", None),
                F=getattr(self.cfg, "async_fan_out", None),
                generator=self.gen,
                noise=(self.parity_noise[f]
                       if self.parity_noise is not None else None))
            self.dbg_toks[f].copy_(toks)
            self.dbg_raws[f].copy_(raws.float())
            # ── 삽입 + [R,NV] 직접 기록.  Probability arithmetic remains
            # in PyTorch to preserve the exact priority values used by the
            # selector; the integer bookkeeping itself is one tiny kernel.
            par = sel.unsqueeze(1).expand(W, C).reshape(-1)
            rq = raws.double().reshape(-1)
            lp = ar.logpri.gather(0, par) + rq.clamp_min(1e-9).log()
            if W > 10 or ar.anc_words > 1:
                _mark_selected_parents_kernel[(triton.cdiv(W, 32),)](
                    sel, sel_valid, ar.cell, ar.state,
                    ROUND=f, W=W, BLOCK=32)
                _insert_tree_children_parallel_kernel[(W * C,)](
                    sel, sel_valid, fan, toks, rq, lp,
                    ar.tok, ar.parent_idx, ar.depth, ar.root, ar.sib,
                    ar.logpri, ar.raw_q, ar.state, ar.cell, ar.valid,
                    ar.anc_bits, ar.n, self._local_idx,
                    self.out_tok, self.out_par, self.out_sib,
                    self.out_rawq, self.out_pcell, self.out_valid,
                    W=W, C=C, NV=NV, ANC_WORDS=ar.anc_words)
                if not _global:
                    _advance_backbone_tips_parallel_kernel[(W,)](
                        sel, sel_valid, fan, ar.root, ar.n,
                        tip_idx, tip_depth, W=W)
                _commit_tree_children_parallel_kernel[(R,)](
                    sel, sel_valid, fan, rq, ar.root,
                    self.out_valid, ar.n, W=W, R=R, C=C)
            else:
                _insert_tree_children_kernel[(1,)](
                    sel, sel_valid, fan, toks, rq, lp,
                    ar.tok, ar.parent_idx, ar.depth, ar.root, ar.sib,
                    ar.logpri, ar.raw_q, ar.state, ar.cell, ar.valid,
                    ar.anc_bits, ar.n, self._local_idx,
                    self.out_tok, self.out_par, self.out_sib, self.out_rawq,
                    self.out_pcell, self.out_valid, tip_idx, tip_depth,
                    ROUND=f, W=W, R=R, C=C, NV=NV,
                    ANC_WORDS=ar.anc_words, GLOBAL=_global)


        if not finalize:
            return
        if os.environ.get("SSD_TREE_EXEC_SKIP_FINALIZE_DIAG", "0") == "1":
            # Diagnostic only: keep the captured forward/tree-update body but
            # make every produced view unservable.  This isolates the two
            # final Triton metadata kernels without allowing incomplete qref
            # metadata to reach the verifier.
            self.out_valid.zero_()
            self.out_u_valid.zero_()
        else:
            self._finalize_outputs()

    def _finalize_outputs(self):
        """Build parent-q and backbone outputs using fixed-shape GPU ops.

        ``build_root_views`` assigns parent-q ids by first occurrence of a
        parent cell, then projects the first-child (sib=0) chain.  This is the
        exact same rule expressed without host readback or data-dependent
        Python loops, so it can live at the tail of the captured graph.
        """
        R, OR, NV, F, V = (self.R, self.out_rows, self.NV,
                            self.F, self.V)
        _finalize_tree_meta_kernel[(R,)](
            self.view_tok, self.view_par, self.view_sib, self.view_pcell,
            self.out_valid, self.out_pq_ref, self.out_pq_cells,
            self.out_u_valid, self.out_backbone_tok,
            self.out_backbone_pcell, NV=NV, F=F)
        # One bounded gather writes directly to the persistent fp16/bf16
        # buffer.  This avoids the former index_select + where path's large
        # float32 temporaries (OOM at P1 K1=9/max_nodes=18).
        _gather_backbone_logits(
            self.cell_logits,
            self.out_backbone_pcell,
            self.out_backbone_logits,
        )

    @torch.inference_mode()
    def capture(self, n_pages0, graph_pool=None):
        # inference_mode 필수 — 실모델 호출이 autograd에 추적되면
        # inplace 충돌 (엔진 실행 경로 규약; 실기 스모크로 확인)
        if n_pages0 not in self.wrappers:
            self.prepare_bucket(n_pages0)
        local_idx = self._local_idx_by_bucket.get(n_pages0)
        if local_idx is None:
            local_idx = torch.full((self.arena.capacity,), -1,
                                   dtype=torch.int64, device=self.dev)
            self._local_idx_by_bucket[n_pages0] = local_idx
        self._local_idx = local_idx
        # 워밍업 ×2 (allocator/커널 준비 — eager)
        for _ in range(2):
            self.run_once(n_pages0, finalize=False)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        g.register_generator_state(self.gen)
        # Multiple page/context variants are mutually exclusive at replay
        # time.  Their captured model intermediates may therefore share one
        # CUDA graph memory pool; persistent inputs/outputs are allocated
        # outside capture and never alias this pool.  Without sharing, P1's
        # two context widths x seven page shapes reserved about 3.1 GiB on a
        # 24 GiB draft GPU and starved the established chain graphs.
        with torch.cuda.graph(g, pool=graph_pool):
            self.run_once(n_pages0, finalize=False)
        self.graphs[n_pages0] = g
        return g

    @torch.inference_mode()
    def replay(self, n_pages0):
        """Replay the four-round graph, then finalize dynamic output rows."""
        self._local_idx = self._local_idx_by_bucket[n_pages0]
        self.graphs[n_pages0].replay()
        if os.environ.get("SSD_TREE_EXEC_SYNC_DIAG", "") == "graph":
            torch.cuda.synchronize()
        self._finalize_outputs()
        if os.environ.get("SSD_TREE_EXEC_SYNC_DIAG", "") == "finalize":
            torch.cuda.synchronize()
