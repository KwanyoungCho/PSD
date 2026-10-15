"""Small, integer-only PS metadata operations; no tree selection or RNG."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _state_kernel(ROPE, SLOTS, CTXS, PAGES, OUT,
                  SS0: tl.constexpr, SS1: tl.constexpr,
                  CS0: tl.constexpr, RS: tl.constexpr, PS: tl.constexpr,
                  F: tl.constexpr, W: tl.constexpr, R: tl.constexpr,
                  P: tl.constexpr, MAX_POS: tl.constexpr,
                  N_SLOTS: tl.constexpr, N_BLOCKS: tl.constexpr,
                  PAGE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    ctx = tl.load(CTXS).to(tl.int64)
    slot = tl.load(SLOTS + (i // W) * SS0 + (i % W) * SS1,
                   i < F * W, other=0).to(tl.int64)
    contexts = tl.load(CTXS + i * CS0, i < F, other=1).to(tl.int64)
    rope = tl.load(ROPE + i * RS, i < R, other=0).to(tl.int64)
    page = tl.load(PAGES + i * PS, i < P, other=0).to(tl.int64)
    # Torch uses floor division, including invalid negative contexts.
    need = (ctx + PAGE_SIZE - 1) // PAGE_SIZE
    bad_rope = tl.sum(((i < R) & ((rope < 0) | (rope >= MAX_POS))).to(tl.int32), 0) > 0
    bad_slot = tl.sum(((i < F * W) & ((slot < 0) | (slot >= N_SLOTS))).to(tl.int32), 0) > 0
    bad_ctx = tl.sum(((i < F) & ((contexts <= 0) | (contexts > MAX_POS))).to(tl.int32), 0) > 0
    bad_page = tl.sum(((i < P) & (i < need) & ((page < 0) | (page >= N_BLOCKS))).to(tl.int32), 0) > 0
    tl.store(OUT, ctx)
    tl.store(OUT + 1, R <= 0)
    tl.store(OUT + 2, ctx <= 0)
    tl.store(OUT + 3, bad_rope)
    tl.store(OUT + 4, bad_slot)
    tl.store(OUT + 5, bad_ctx)
    tl.store(OUT + 6, bad_page)


def state_snapshot(rope, slots, contexts, pages, out, *, F, W, R,
                   max_pos, n_slots, n_blocks, block_size):
    """Same six validity flags as DraftRunner's reference, one GPU launch.

    The caller owns ``out`` and reads it once before dispatch. All W lanes
    require valid slots; only the R real roots require valid RoPE positions.
    Pages beyond the live prefix remain deliberately unchecked.
    """
    _state_kernel[(1,)](
        rope, slots, contexts, pages, out,
        slots.stride(0), slots.stride(1), contexts.stride(0),
        rope.stride(0), pages.stride(1), F, W, R, pages.shape[1],
        max_pos, n_slots, n_blocks, block_size,
        triton.next_power_of_2(max(F * W, R, pages.shape[1])),
        num_warps=4)
    return out.cpu().tolist()


def copy_glue_rows(executor, rows, roots, width):
    """Keep a GPU-produced ancestry mask on GPU; retain host diagnostics."""
    if torch.is_tensor(rows):
        executor.in_glue[:roots, :width].copy_(rows[:roots, :width])
    else:
        np.copyto(executor.in_glue_host[:roots, :width].numpy(),
                  np.ascontiguousarray(rows[:roots, :width]))
        executor.in_glue[:roots, :width].copy_(
            executor.in_glue_host[:roots, :width], non_blocking=True)


def host_glue_rows(rows):
    """Materialize only for reference/diagnostic backends requiring NumPy."""
    if torch.is_tensor(rows):
        return rows.detach().cpu().numpy().astype(np.uint8, copy=False)
    return rows


def host_fanout(layout):
    """Materialize deferred fan counts only for an eager/diagnostic consumer.

    The captured tree executor needs the GPU fan map and the known root
    count, not these Python counts. Never retain an earlier step's list.
    """
    if layout.fan_out_list is None:
        layout.fan_out_list = layout.fan_out_t.tolist()
        layout.fan_out_list_miss = layout.fan_out_list
    return layout.fan_out_list
