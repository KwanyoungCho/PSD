"""Optional integer-only response packing; retain host validation snapshot.

The immutable device packet is assembled alongside the topology, before the
single host readback. No sampling, pruning, residual checks, or RNG changes.
"""
import torch
import triton
import triton.language as tl

from ssd.engine.helpers.response_topology import ResponseTopology
from ssd.engine.helpers.p2_tree import tree_wire_ints_len


@triton.jit
def _pack(TOK, PAR, SIB, REF, VALID, UVALID, OUT, PACK,
          T0: tl.constexpr, T1: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr,
          VS: tl.constexpr, US: tl.constexpr, CAP: tl.constexpr,
          COPY: tl.constexpr, WIDTH: tl.constexpr, PHASE: tl.constexpr,
          WIRE: tl.constexpr, PACKLEN: tl.constexpr, BLOCK: tl.constexpr):
    root = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    slot = (i - 3) % CAP
    field = (i - 3) // CAP
    valid = tl.load(VALID + root * VS).to(tl.int64)
    uvalid = tl.load(UVALID + root * US).to(tl.int64)
    live = (i >= 3) & (i < WIRE) & (slot < COPY)
    tok = tl.load(TOK + root * T0 + slot * T1,
                  live & (field == 0), other=0).to(tl.int64)
    par = tl.load(PAR + root * P0 + slot * P1,
                  live & (field == 1), other=0).to(tl.int64)
    sib = tl.load(SIB + root * S0 + slot * S1,
                  live & (field == 2), other=0).to(tl.int64)
    ref = tl.load(REF + root * Q0 + slot * Q1,
                  live & (field == 3), other=0).to(tl.int64)
    val = tl.where(i == 0, valid, tl.where(i == 1, uvalid,
                   tl.where(i == 2, 1, tok + par + sib + ref)))
    tl.store(OUT + root * WIRE + i, val, i < WIRE)
    tl.store(PACK + root * PACKLEN + 3 + WIDTH + i, val, i < WIRE)
    packet_tok = tl.load(TOK + root * T0 + i * T1,
                         i < COPY, other=0).to(tl.int64)
    tl.store(PACK + root * PACKLEN + 3 + i, packet_tok, i < WIDTH)
    tl.store(PACK + root * PACKLEN, 1)
    tl.store(PACK + root * PACKLEN + 1, PHASE)
    tl.store(PACK + root * PACKLEN + 2, valid)


def prepare_response_topology_packet(views, capacity, phase, width):
    count, nodes = views['tok'].shape
    if width < capacity:
        raise ValueError('Response token width must cover topology capacity')
    wire = tree_wire_ints_len(capacity)
    if wire != 3 + 4 * capacity or capacity < 1:
        raise ValueError('Unsupported topology wire schema')
    device = views['valid'].device
    packed = torch.empty((count, wire), dtype=torch.int64, device=device)
    packet = torch.empty((count, 3 + width + wire), dtype=torch.int64, device=device)
    if count:
        tok, par, sib, ref = (views[k] for k in
                             ('tok', 'parent_local', 'sib_order', 'parent_q_ref'))
        _pack[(count,)](
            tok, par, sib, ref, views['valid'], views['u_valid'], packed, packet,
            *tok.stride(), *par.stride(), *sib.stride(), *ref.stride(),
            views['valid'].stride(0), views['u_valid'].stride(0),
            capacity, min(nodes, capacity), width, phase, wire, packet.shape[1],
            triton.next_power_of_2(max(wire, width)), num_warps=4)
    # Keep the exact immutable host data used by existing validation and glue.
    host = packed.detach().cpu()
    integers = tuple(tuple(row) for row in host.tolist())
    return ResponseTopology(capacity, packed, host, integers), packet
