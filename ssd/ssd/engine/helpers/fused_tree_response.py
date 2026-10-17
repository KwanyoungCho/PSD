"""B1 tree response path using prebuilt topology and one readback.

No sampling, root, pruning, or verification rules change. The discarded
uniform q allocation intentionally preserves the original RNG consumption.
"""
import os
import numpy as np
import torch


def attach_response_packet(snapshot, phase, width):
    count = len(snapshot.integer_rows)
    packet = np.zeros((count, 3 + width + snapshot.device_rows.shape[1]), dtype=np.int64)
    packet[:, 0] = 1
    packet[:, 1] = phase
    for root, row in enumerate(snapshot.integer_rows):
        packet[root, 2] = row[0]
        packet[root, 3:3 + snapshot.capacity] = row[3:3 + snapshot.capacity]
        packet[root, 3 + width:] = row
    # Kept alongside the topology snapshot, whose executor lifetime is already
    # bounded by the next request. No live buffer is reused before send.
    return torch.from_numpy(packet).to(snapshot.device_rows.device)


def try_response(r, keys, batch, k, num_tokens):
    if batch != 1 or r.config.use_eagle or not r.config.duet_enabled:
        return None
    if r.tree_cache_keys.numel() == 0:
        return None
    # A chain-only or G>M cache has no prebuilt tree response. Avoid an
    # additional lookup/readback before its established serving path.
    if not any(views is not None and views.get('_response_packet') is not None
               for views in (getattr(r, '_p1_tree_views', None),
                             getattr(r, '_tree_views', None))):
        return None
    if r.config.verbose or r.config.duet_proxy_on_draft:
        return None
    if any(os.environ.get(x, '') not in ('', '0') for x in (
            'SSD_TREE_STAGE2', 'SSD_TREE_ROOT_SHADOW', 'SSD_TREE_TOPO_TRACE')):
        return None
    # All host decisions share one bounded transfer, instead of individually
    # synchronizing hit/all/phase/root/valid/sequence/context scalar reads.
    match = (keys[0] == r.tree_cache_keys).all(dim=1)
    hit, index = match.to(torch.int64).max(dim=0)
    desc = torch.cat((keys.reshape(-1), num_tokens.reshape(-1),
                      hit.reshape(1), index.reshape(1))).tolist()
    seq, terminal, recovery, ntok, hit, index = desc
    if not hit:
        r._fused_miss_match = match.view(1, -1)
        return None
    phase = 1 if index < r._last_n_draft_keys else 2
    views = getattr(r, '_p1_tree_views' if phase == 1 else '_tree_views', None)
    if views is None:
        return None
    snapshot = views.get('_response_topology')
    packet = views.get('_response_packet')
    if snapshot is None or packet is None:
        return None
    root = index if phase == 1 else index - r._last_n_draft_keys
    packed, packed_cpu, parsed = snapshot.select(root, r.hf_config.vocab_size)
    nv = int(parsed['valid'])
    if nv <= 0:
        return None
    width = int(r.config.duet_response_token_width)
    response = packet[root]
    # Keep exactly the original random generator transition, even though a
    # tree hit never consumes the dummy ordinary-q scores or their argmax.
    out_logits = torch.empty((1, k, r.hf_config.vocab_size),
        dtype=r.hf_config.torch_dtype, device=r.device).uniform_()
    from ssd.engine.helpers.p2_tree_executor import _gather_backbone_logits
    from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close
    label = f'tree_hit_parent_q_gather_p{phase}'
    event = duet_record(label)
    cells = views['parent_q_cells'][root, :nv].to(
        device=views['cell_logits'].device, dtype=torch.int64)
    _gather_backbone_logits(views['cell_logits'], cells,
                           r._tree_wire_parent_q_buf[0, :nv])
    duet_close(label, event)
    r._tree_wire_ints = packed.to(r.device)
    r._tree_wire_parent_q = r._tree_wire_parent_q_buf
    r._tree_hit_root, r._tree_hit_views = root, views
    r._tree_hit_phase = phase
    r._tree_hit_nv = (r.config.duet_p1_tree_verify_nodes if phase == 1
                      else r.config.duet_p2_tree_verify_nodes)
    r._tree_served_ints = packed_cpu.clone()
    r._tree_served_numtok, r._tree_served_seq = ntok, seq
    r._fused_tree_response = response
    r._tree_hit_parsed = parsed
    # Glue is not needed to send verification inputs. Construct it after
    # the response is sent, before the next draft operation consumes it.
    tokens = response[3:3 + width].view(1, width)
    status = f'hit_k{phase}' if os.environ.get('SSD_PROFILE_DUET', '0') == '1' else None
    return (tokens, out_logits, None, response[:1], None,
            response[1:2], response[2:3], nv, status)


def glue_context(parents):
    """Build tiny ancestry metadata on CPU, then copy each array once."""
    n = len(parents)
    depths = [0] * n
    ancestors = [[] for _ in range(n)]
    rows = np.zeros((n + 1, n + 1), dtype=np.uint8)
    rows[0, 0] = 1
    ctx_depth = np.zeros(n + 1, dtype=np.int64)
    for j, p in enumerate(parents):
        if p >= 0:
            depths[j] = depths[p] + 1
            ancestors[j] = ancestors[p] + [p]
        rows[j + 1, 0] = rows[j + 1, j + 1] = 1
        rows[j + 1, 1 + np.asarray(ancestors[j], dtype=np.int64)] = 1
        ctx_depth[j + 1] = depths[j] + 1
    return depths, ancestors, rows, ctx_depth


def request_packet_cpu(seqs, max_blocks):
    """Identical integer wire packet, prepared in one host allocation."""
    b = len(seqs)
    packet = np.zeros(5 * b + b * max_blocks, dtype=np.int64)
    keys = packet[:3*b].reshape(b, 3)
    lengths = packet[3*b:4*b]
    tables = packet[4*b:4*b+b*max_blocks].reshape(b, max_blocks)
    tables.fill(-1)
    temps = np.empty(b, dtype=np.float32)
    for i, seq in enumerate(seqs):
        terminal = getattr(seq, 'tree_terminal_node', None)
        keys[i] = (seq.seq_id, terminal if terminal is not None
                   else seq.last_spec_step_accepted_len-1, seq.recovery_token_id)
        lengths[i] = seq.num_tokens
        bt = seq.draft_block_table
        if len(bt) > max_blocks:
            raise ValueError('Draft block table exceeds request capacity')
        tables[i, :len(bt)] = bt
        temps[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
    packet[-b:] = temps.view(np.int32).astype(np.int64)
    return torch.from_numpy(packet)
