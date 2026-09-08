"""Fused CUDA reranking for DUET's tiny P1 candidate trees.

The scalar reference lives in :mod:`ssd.engine.helpers.p2_tree`.  This module
only changes where that fixed policy executes: one Triton program handles one
root and writes the already-compacted response into persistent buffers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rerank_tree_kernel(
    tok_ptr, par_ptr, sib_ptr, rawq_ptr, pqref_ptr, pqcells_ptr, valid_ptr,
    out_tok_ptr, out_par_ptr, out_sib_ptr, out_rawq_ptr,
    out_pqref_ptr, out_pqcells_ptr, out_valid_ptr, out_uvalid_ptr,
    packed_ptr,
    GENERATED: tl.constexpr,
    VERIFY: tl.constexpr,
    WIRE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One scalar program per root; all widths are compile-time tiny."""
    r = tl.program_id(0)
    src_base = r * GENERATED
    out_base = r * WIRE
    packed_width: tl.constexpr = 3 + 4 * WIRE
    packed_base = r * packed_width
    n = tl.load(valid_ptr + r)

    offs = tl.arange(0, BLOCK)
    lane = offs < GENERATED
    parent_v = tl.load(par_ptr + src_base + offs, mask=lane, other=-1) \
        .to(tl.int64)
    sibling_v = tl.load(sib_ptr + src_base + offs, mask=lane, other=0) \
        .to(tl.int64)
    raw_v = tl.load(rawq_ptr + src_base + offs, mask=lane, other=0.0) \
        .to(tl.float64)
    live_v = lane & (offs < n)
    positive_finite = (raw_v > 0.0) & (raw_v < float("inf"))
    q_v = tl.where(live_v & positive_finite, raw_v, 0.0)

    # Cumulative path confidence and prerequisite closure bitsets.  A node's
    # closure contains all ancestors plus preceding ordered siblings at every
    # level, exactly as rerank_tree_indices._closure does.
    path_v = tl.zeros((BLOCK,), tl.float64)
    closure_v = tl.zeros((BLOCK,), tl.int64)
    bit_v = tl.full((BLOCK,), 1, tl.int64) << offs
    for j in tl.static_range(0, GENERATED):
        p = tl.sum(tl.where(offs == j, parent_v, 0))
        qj = tl.sum(tl.where(offs == j, q_v, 0.0))
        parent_conf = tl.sum(tl.where(offs == p, path_v, 0.0))
        conf = qj * tl.where(p < 0, 1.0, parent_conf)
        path_v = tl.where(offs == j, conf, path_v)

        sj = tl.sum(tl.where(offs == j, sibling_v, 0))
        group = (
            live_v & (parent_v == p) & (sibling_v <= sj)
            & (j < n)
        )
        group_bits = tl.sum(tl.where(group, bit_v, 0))
        parent_bits = tl.sum(tl.where(offs == p, closure_v, 0))
        closure = group_bits | tl.where(p >= 0, parent_bits, 0)
        closure_v = tl.where(offs == j, closure, closure_v)

    selected = tl.full((), 0, tl.int64)
    processed = tl.full((), 0, tl.int64)
    for _ in tl.static_range(0, GENERATED):
        best = tl.full((), -float("inf"), tl.float64)
        best_j = tl.full((), -1, tl.int64)
        for j in tl.static_range(0, GENERATED):
            score = tl.sum(tl.where(offs == j, path_v, 0.0))
            unseen = ((processed >> j) & 1) == 0
            take = (j < n) & unseen & (score > best)
            best = tl.where(take, score, best)
            best_j = tl.where(take, j, best_j)
        safe_best = tl.maximum(best_j, 0)
        need = tl.sum(tl.where(offs == safe_best, closure_v, 0))
        expanded = selected | need
        count = tl.full((), 0, tl.int64)
        for k in tl.static_range(0, GENERATED):
            count += (expanded >> k) & 1
        fits = (best_j >= 0) & (count <= VERIFY)
        selected = tl.where(fits, expanded, selected)
        processed = processed | tl.where(
            best_j >= 0, tl.full((), 1, tl.int64) << safe_best, 0)

    # Generation-order best-effort fill from the scalar reference.
    for j in tl.static_range(0, GENERATED):
        need = tl.sum(tl.where(offs == j, closure_v, 0))
        expanded = selected | need
        old_count = tl.full((), 0, tl.int64)
        new_count = tl.full((), 0, tl.int64)
        for k in tl.static_range(0, GENERATED):
            old_count += (selected >> k) & 1
            new_count += (expanded >> k) & 1
        fits = (j < n) & (old_count < VERIFY) & (new_count <= VERIFY)
        selected = tl.where(fits, expanded, selected)

    # Deterministic padding for both the compact view and response wire.
    for j in tl.static_range(0, WIRE):
        tl.store(out_tok_ptr + out_base + j, 0)
        tl.store(out_par_ptr + out_base + j, -1)
        tl.store(out_sib_ptr + out_base + j, 0)
        tl.store(out_rawq_ptr + out_base + j, 0.0)
        tl.store(out_pqref_ptr + out_base + j, -1)
        tl.store(out_pqcells_ptr + out_base + j, -1)
        tl.store(packed_ptr + packed_base + 3 + j, 0)
        tl.store(packed_ptr + packed_base + 3 + WIRE + j, -1)
        tl.store(packed_ptr + packed_base + 3 + 2 * WIRE + j, 0)
        tl.store(packed_ptr + packed_base + 3 + 3 * WIRE + j, -1)

    compact_ref_v = tl.full((BLOCK,), -1, tl.int64)
    out_n = tl.full((), 0, tl.int64)
    unique_n = tl.full((), 0, tl.int64)
    for j in tl.static_range(0, GENERATED):
        take = ((selected >> j) & 1) != 0
        p = tl.sum(tl.where(offs == j, parent_v, 0))
        compact_parent = tl.full((), 0, tl.int64)
        for k in tl.static_range(0, j):
            compact_parent += (((selected >> k) & 1) & (k < p))
        compact_parent = tl.where(p < 0, -1, compact_parent)

        ref = tl.load(pqref_ptr + src_base + j)
        seen = tl.full((), 0, tl.int1)
        compact_ref = unique_n
        for k in tl.static_range(0, j):
            prior_ref = tl.load(pqref_ptr + src_base + k)
            same = (((selected >> k) & 1) != 0) & (prior_ref == ref)
            prior_compact = tl.sum(tl.where(
                offs == k, compact_ref_v, 0))
            compact_ref = tl.where(same, prior_compact, compact_ref)
            seen = seen | same
        is_new = take & ~seen
        compact_ref_v = tl.where(
            offs == j, tl.where(take, compact_ref, -1), compact_ref_v)

        dst = out_base + out_n
        packed_dst = packed_base + 3 + out_n
        tok = tl.load(tok_ptr + src_base + j)
        sib = tl.sum(tl.where(offs == j, sibling_v, 0))
        raw = tl.load(rawq_ptr + src_base + j)
        tl.store(out_tok_ptr + dst, tok, mask=take)
        tl.store(out_par_ptr + dst, compact_parent, mask=take)
        tl.store(out_sib_ptr + dst, sib, mask=take)
        tl.store(out_rawq_ptr + dst, raw, mask=take)
        tl.store(out_pqref_ptr + dst, compact_ref, mask=take)
        tl.store(packed_ptr + packed_dst, tok, mask=take)
        tl.store(packed_ptr + packed_dst + WIRE, compact_parent, mask=take)
        tl.store(packed_ptr + packed_dst + 2 * WIRE, sib, mask=take)
        tl.store(packed_ptr + packed_dst + 3 * WIRE, compact_ref, mask=take)

        source_cell = tl.load(
            pqcells_ptr + src_base + tl.maximum(ref, 0),
            mask=is_new, other=-1)
        tl.store(out_pqcells_ptr + out_base + unique_n,
                 source_cell, mask=is_new)
        unique_n += is_new.to(tl.int64)
        out_n += take.to(tl.int64)

    tl.store(out_valid_ptr + r, out_n)
    tl.store(out_uvalid_ptr + r, unique_n)
    tl.store(packed_ptr + packed_base, out_n)
    tl.store(packed_ptr + packed_base + 1, unique_n)
    tl.store(packed_ptr + packed_base + 2, 1)


@torch.inference_mode()
def precompute_reranked_tree_views_fused_gpu(
    token: torch.Tensor,
    parent_local: torch.Tensor,
    sibling_order: torch.Tensor,
    raw_q: torch.Tensor,
    parent_q_ref: torch.Tensor,
    parent_q_cells: torch.Tensor,
    valid: torch.Tensor,
    *,
    verify_cap: int,
    wire_cap: int,
    output_buffers: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Launch the exact fused rerank into persistent output buffers."""
    if token.ndim != 2 or token.device.type != "cuda":
        raise ValueError("fused P1 rerank token must be a CUDA [R,N] tensor")
    roots, generated_cap = token.shape
    inputs = (
        parent_local, sibling_order, raw_q, parent_q_ref, parent_q_cells)
    if any(x.device != token.device or x.shape != token.shape for x in inputs):
        raise ValueError("fused P1 rerank input shape/device mismatch")
    if valid.device != token.device or valid.shape != (roots,):
        raise ValueError("fused P1 rerank valid shape/device mismatch")
    verify_cap = int(verify_cap)
    wire_cap = int(wire_cap)
    if not (1 <= verify_cap <= generated_cap <= 62
            and verify_cap <= wire_cap <= 62):
        raise ValueError(
            "fused P1 rerank requires 1 <= verify <= generated <= 62 and "
            "verify <= wire <= 62; "
            f"got {verify_cap}, {generated_cap}, {wire_cap}")
    required = (
        "tok", "parent_local", "sib_order", "raw_q", "parent_q_ref",
        "parent_q_cells", "valid", "u_valid", "packed")
    for name in required:
        if name not in output_buffers:
            raise ValueError(f"missing fused P1 rerank output buffer {name}")
    out = {name: output_buffers[name][:roots] for name in required}
    if any(out[name].shape != (roots, wire_cap) for name in required[:6]):
        raise ValueError("fused P1 rerank view output shape mismatch")
    if out["valid"].shape != (roots,) or out["u_valid"].shape != (roots,):
        raise ValueError("fused P1 rerank count output shape mismatch")
    if out["packed"].shape != (roots, 3 + 4 * wire_cap):
        raise ValueError("fused P1 rerank packed output shape mismatch")

    block = triton.next_power_of_2(generated_cap)
    _rerank_tree_kernel[(roots,)](
        token, parent_local, sibling_order, raw_q,
        parent_q_ref, parent_q_cells, valid,
        out["tok"], out["parent_local"], out["sib_order"], out["raw_q"],
        out["parent_q_ref"], out["parent_q_cells"],
        out["valid"], out["u_valid"], out["packed"],
        GENERATED=generated_cap, VERIFY=verify_cap, WIRE=wire_cap,
        BLOCK=block, num_warps=1,
    )
    return out
