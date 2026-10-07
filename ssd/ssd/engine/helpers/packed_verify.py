"""Pack real chain queries into bounded CUDA-graph token buckets.

The extra sequence owns all alignment queries. Its KV writes are disabled;
real sequences keep exact query/context lengths, so causal alignment never
changes when a neighbour has a shorter proposal.
"""
import torch
from ssd.utils.context import set_context


def packed_shape(batch, tokens, buckets, max_k):
    cap = next(b for b in buckets if b >= batch)
    return cap, min(((tokens + 3) // 4) * 4, cap * (max_k + 1))


def build_packed_layout(seqs, valid_k, dense_k, block_size, batch_capacity,
                        token_capacity):
    if len(seqs) != len(valid_k) or not seqs:
        raise ValueError("Packed verify needs one length per live sequence")
    ids, positions, slots, lens, dense_rows = [], [], [], [], []
    cu = [0]
    for seq, k in zip(seqs, valid_k):
        if not 0 <= k <= dense_k:
            raise ValueError(f"Invalid packed proposal length {k}/{dense_k}")
        start = seq.num_tokens - dense_k - 1
        if start != seq.num_cached_tokens:
            raise AssertionError("Packed verify window is not at the cached frontier")
        offset = len(ids)
        ids.extend(seq.token_ids[start:start+k+1])
        positions.extend(range(start, start+k+1))
        slots.extend(seq.block_table[p//block_size]*block_size+p%block_size
                     for p in range(start, start+k+1))
        lens.append(start+k+1)
        dense_rows.extend(offset+j if j <= k else offset for j in range(dense_k+1))
        cu.append(len(ids))
    if len(seqs) > batch_capacity or len(ids) > token_capacity:
        raise ValueError("Packed graph capacity too small")
    # Empty real lanes, followed by a distinct alignment sequence.
    cu.extend([len(ids)] * (batch_capacity-len(seqs)))
    cu.append(token_capacity)
    lens.extend([1] * (batch_capacity+1-len(seqs)))
    return dict(input_ids=ids, positions=positions, slot_mapping=slots,
                context_lens=lens, cu_seqlens_q=cu, dense_rows=dense_rows)


def prepare_packed_verify(runner, seqs, valid_k, dense_k):
    buckets = runner.graph_bs_list["duet_verify_k1"]
    shape = packed_shape(len(seqs), sum(k+1 for k in valid_k), buckets,
                         runner.config.duet_phase1_k)
    layout = build_packed_layout(seqs, valid_k, dense_k, runner.block_size, *shape)
    dev = runner.device
    def tensor(x, dtype=torch.int64):
        return torch.tensor(x, dtype=dtype, pin_memory=True).to(dev, non_blocking=True)
    table = [s.block_table + [s.block_table[0]]*(runner.max_num_blocks-len(s.block_table))
             for s in seqs]
    table += [table[0]] * (shape[0]+1-len(seqs))
    runner._duet_packed_meta = dict(shape=shape, valid_k=valid_k,
                                   dense_rows=tensor(layout["dense_rows"]))
    set_context(False,
        cu_seqlens_q=tensor(layout["cu_seqlens_q"], torch.int32),
        max_seqlen_q=runner.config.duet_phase1_k+1,
        slot_mapping=tensor(layout["slot_mapping"], torch.int32),
        context_lens=tensor(layout["context_lens"], torch.int32),
        block_tables=tensor(table, torch.int32))
    return tensor(layout["input_ids"]), tensor(layout["positions"])


def stage_packed_graph(gv, ids, positions, context):
    live = ids.numel()
    gv["input_ids"].zero_()
    gv["positions"].zero_()
    gv["slot_mapping"].fill_(-1)
    gv["input_ids"][:live].copy_(ids)
    gv["positions"][:live].copy_(positions)
    gv["slot_mapping"][:live].copy_(context.slot_mapping)
    gv["context_lens"].copy_(context.context_lens)
    gv["cu_seqlens_q"].copy_(context.cu_seqlens_q)
    gv["block_tables"].copy_(context.block_tables)


def capture_packed_graphs(runner):
    from ssd.engine.helpers.cudagraph_helpers import capture_duet_verify_cudagraph
    from ssd.layers.fi_attn import use_flashinfer_attention
    if use_flashinfer_attention(runner.device):
        raise ValueError("Packed chain verification currently requires the SGL attention backend")
    k1, k2 = runner.config.duet_phase1_k, runner.config.duet_phase2_k
    buckets = runner.graph_bs_list["duet_verify_k1"]
    shapes = set()
    for b in range(2, runner.config.max_num_seqs+1):
        for nlong in range(b+1):
            shapes.add(packed_shape(b, b*(k2+1)+nlong*(k1-k2), buckets, k1))
        # Mixed-hit AR fallback can give miss rows zero proposals. Capture
        # all resulting token capacities without specializing row patterns.
        import os
        if os.environ.get("SSD_MIXED_MISS_AR", "0") == "1":
            for tokens in range(b+k2, b*(k1+1)+1):
                shapes.add(packed_shape(b, tokens, buckets, k1))
    pool = None
    runner._duet_packed_buckets = {}
    for shape in sorted(shapes, reverse=True):
        name = f"duet_packed_{shape[0]}_{shape[1]}"
        gv, pool, pre, post, bs = capture_duet_verify_cudagraph(
            runner, lookahead=k1, graph_pool=pool, packed_shape=shape)
        runner.graph_vars[name] = gv
        runner.graphs[name+"_pre"] = pre
        runner.graphs[name+"_post"] = post
        runner.graph_bs_list[name] = bs
        runner.graph_pools[name] = pool
        runner._duet_packed_buckets[shape] = name
    print(f"[DUET] Captured {len(shapes)} packed chain graph shapes", flush=True)
