"""GPU preparation kernels for DUET target-side dynamic-tree metadata.

The dynamic-tree wire is tiny, but the old serving path expanded it through
Python lists, freshly allocated CPU tensors, NumPy mask packing, and several
small H2D copies on every cache hit.  These kernels keep the same wire and
verification contracts while writing directly into persistent CUDA buffers.

CPU parsing/validation deliberately remains at the wire boundary for now.  It
is the coherent-failure guard for all target TP ranks; this module only
replaces the repeated *derived topology* construction after that validation.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_proxy_topology_kernel(
    par_ptr,
    sib_ptr,
    child_ptr,
    child_valid_ptr,
    par_out_ptr,
    sib_out_ptr,
    node_valid_ptr,
    NV: tl.constexpr,
    C_MAX: tl.constexpr,
):
    """Pack one validated parent/sibling prefix into fixed graph buffers."""
    # NV is at most a few dozen in the supported DUET configurations.  One
    # scalar program avoids launching reset/scatter kernels for each field.
    for flat in tl.static_range(0, (NV + 1) * C_MAX):
        tl.store(child_ptr + flat, NV)
        tl.store(child_valid_ptr + flat, 0)
    for j in tl.static_range(0, NV):
        tl.store(par_out_ptr + j, -1)
        tl.store(sib_out_ptr + j, 0)
        tl.store(node_valid_ptr + j, 0)

    for j in tl.static_range(0, NV):
        parent = tl.load(par_ptr + j)
        sibling = tl.load(sib_ptr + j)
        dst = (parent + 1) * C_MAX + sibling
        tl.store(child_ptr + dst, j)
        tl.store(child_valid_ptr + dst, 1)
        tl.store(par_out_ptr + j, parent)
        tl.store(sib_out_ptr + j, sibling)
        tl.store(node_valid_ptr + j, 1)


@torch.inference_mode()
def pack_tree_proxy_topology_gpu_(
    parent: torch.Tensor,
    sibling: torch.Tensor,
    topology: dict[str, torch.Tensor],
) -> None:
    """Write a validated exact-width topology into persistent GPU buffers.

    ``topology`` is the dictionary owned by :class:`TreeProxyCUDAGraph`.
    The graph bucket width equals the live node count, so every source row is
    valid and no device scalar readback is needed.
    """
    required = {"child", "child_valid", "par", "sib", "node_valid"}
    if set(topology) != required:
        raise ValueError(
            f"unexpected tree proxy topology fields: {sorted(topology)}")
    child = topology["child"]
    child_valid = topology["child_valid"]
    par_out = topology["par"]
    sib_out = topology["sib"]
    node_valid = topology["node_valid"]
    nv = int(par_out.numel())
    if parent.device.type != "cuda" or sibling.device.type != "cuda":
        raise ValueError("GPU topology packing requires CUDA parent/sibling")
    if parent.device != par_out.device or sibling.device != par_out.device:
        raise ValueError("tree topology source/output devices differ")
    if parent.numel() < nv or sibling.numel() < nv:
        raise ValueError(
            f"tree topology source shorter than bucket: parent={parent.numel()} "
            f"sibling={sibling.numel()} nv={nv}")
    if child.ndim != 2 or child.shape[0] != nv + 1:
        raise ValueError(
            f"tree child buffer shape mismatch: {tuple(child.shape)} nv={nv}")
    c_max = int(child.shape[1])
    _pack_proxy_topology_kernel[(1,)](
        parent.contiguous(),
        sibling.contiguous(),
        child,
        child_valid,
        par_out,
        sib_out,
        node_valid,
        NV=nv,
        C_MAX=c_max,
        num_warps=1,
    )


@triton.jit(
    do_not_specialize=(
        "pos0", "prefix_len", "kv_len", "mask_bytes"),
    do_not_specialize_on_alignment=(
        "pos0", "prefix_len", "kv_len", "mask_bytes"),
)
def _pack_verify_inputs_kernel(
    par_ptr,
    input_src_ptr,
    slot_src_ptr,
    input_dst_ptr,
    rope_dst_ptr,
    slot_dst_ptr,
    context_len_dst_ptr,
    packed_mask_ptr,
    pos0,
    prefix_len,
    kv_len,
    mask_bytes,
    VALID: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build padded rows, depth RoPE, and the packed tree mask together."""
    pid = tl.program_id(0)

    # Only the first mask program writes the few target query rows.  Folding
    # this into the mask launch removes the second small-kernel dispatch that
    # otherwise erased the savings over the CPU fallback.
    row = tl.arange(0, ROW_BLOCK)
    in_bucket = (pid == 0) & (row < ROWS)
    live = row < VALID + 1
    tl.store(
        input_dst_ptr + row,
        tl.load(input_src_ptr + row, mask=live, other=0),
        mask=in_bucket,
    )
    tl.store(
        slot_dst_ptr + row,
        tl.load(slot_src_ptr + row, mask=live, other=-1),
        mask=in_bucket,
    )
    node_row = row - 1
    row_active = (row > 0) & live
    cur_row = node_row
    depth = tl.zeros((ROW_BLOCK,), tl.int64)
    for _ in tl.static_range(0, VALID):
        parent_row = tl.load(
            par_ptr + cur_row,
            mask=row_active & (cur_row >= 0) & (cur_row < VALID),
            other=-1,
        )
        has_parent = row_active & (parent_row >= 0)
        depth += has_parent.to(tl.int64)
        cur_row = tl.where(has_parent, parent_row, -1)
    rope = tl.where(row_active, pos0 + 1 + depth, pos0)
    tl.store(rope_dst_ptr + row, rope, mask=in_bucket)
    tl.store(context_len_dst_ptr + pid, kv_len, mask=pid == 0)

    byte = pid * BLOCK + tl.arange(0, BLOCK)
    byte_live = byte < mask_bytes
    value = tl.zeros((BLOCK,), tl.int32)
    total_bits = ROWS * kv_len

    for bit in tl.static_range(0, 8):
        flat = byte * 8 + bit
        in_range = byte_live & (flat < total_bits)
        row = flat // kv_len
        col = flat - row * kv_len
        visible = in_range & (col < prefix_len)

        # Row zero is the recovery/context row.  Tree node j occupies row j+1
        # and KV column prefix_len+j.  Padding rows see the prefix only.
        node = row - 1
        tree_col = col - prefix_len
        tree_cell = (
            in_range
            & (row > 0)
            & (row <= VALID)
            & (tree_col >= 0)
            & (tree_col < VALID)
        )
        on_path = tree_cell & (tree_col == node)
        cur = node
        active = tree_cell
        for _ in tl.static_range(0, VALID):
            parent = tl.load(
                par_ptr + cur,
                mask=active & (cur >= 0) & (cur < VALID),
                other=-1,
            )
            on_path = on_path | (active & (parent == tree_col))
            active = active & (parent >= 0)
            cur = tl.where(active, parent, -1)
        visible = visible | (tree_cell & on_path)
        value += visible.to(tl.int32) << bit

    tl.store(packed_mask_ptr + byte, value, mask=byte_live)


@torch.inference_mode()
def build_tree_verify_inputs_gpu_(
    parent: torch.Tensor,
    input_ids: torch.Tensor,
    slot_mapping: torch.Tensor,
    out_input_ids: torch.Tensor,
    out_rope: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    out_context_lens: torch.Tensor,
    out_packed_mask: torch.Tensor,
    *,
    valid: int,
    rows: int,
    pos0: int,
    prefix_len: int,
    kv_len: int,
) -> int:
    """Prepare padded verify rows, RoPE positions, and mask on the GPU.

    Returns the exact number of bytes consumed by the packed mask.  All output
    tensors are persistent buffers owned by the selected target CUDA graph.
    """
    valid = int(valid)
    rows = int(rows)
    pos0 = int(pos0)
    prefix_len = int(prefix_len)
    kv_len = int(kv_len)
    if valid < 1 or rows < valid + 1:
        raise ValueError(f"invalid tree verify shape: valid={valid} rows={rows}")
    if kv_len != prefix_len + valid:
        raise ValueError(
            f"tree KV contract requires kv_len=prefix+valid; got "
            f"{kv_len}!={prefix_len}+{valid}")
    tensors = (
        parent,
        input_ids,
        slot_mapping,
        out_input_ids,
        out_rope,
        out_slot_mapping,
        out_context_lens,
        out_packed_mask,
    )
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("GPU tree verify preparation requires CUDA tensors")
    device = out_input_ids.device
    if any(t.device != device for t in tensors):
        raise ValueError("tree verify preparation tensors are on different GPUs")
    if parent.numel() < valid:
        raise ValueError("tree parent buffer is shorter than valid node count")
    if input_ids.numel() < valid + 1 or slot_mapping.numel() < valid + 1:
        raise ValueError("tree verify source rows are shorter than valid+1")
    if (out_input_ids.numel() < rows or out_rope.numel() < rows
            or out_slot_mapping.numel() < rows):
        raise ValueError("tree verify output buffers are shorter than row bucket")
    if out_context_lens.numel() < 1:
        raise ValueError("tree verify context-length buffer is empty")
    mask_bytes = (rows * kv_len + 7) // 8
    if out_packed_mask.numel() < mask_bytes:
        raise ValueError(
            f"packed mask capacity {out_packed_mask.numel()} < {mask_bytes}")

    mask_block = 256
    _pack_verify_inputs_kernel[(triton.cdiv(mask_bytes, mask_block),)](
        parent.contiguous(),
        input_ids,
        slot_mapping,
        out_input_ids,
        out_rope,
        out_slot_mapping,
        out_context_lens,
        out_packed_mask,
        pos0,
        prefix_len,
        kv_len,
        mask_bytes,
        VALID=valid,
        ROWS=rows,
        ROW_BLOCK=triton.next_power_of_2(rows),
        BLOCK=mask_block,
        num_warps=4,
    )
    return mask_bytes
