"""Request-local contracts for batched DUET trees."""
import os
import numpy as np
from ssd.engine.helpers.tree_host_topology import context_topology


def enabled(config):
    return (getattr(config, 'duet_enabled', False)
            and getattr(config, 'duet_tree_enabled', False)
            and (int(config.max_num_seqs) > 1
                 or os.environ.get('SSD_BATCHED_TREE', '0') == '1'))


def capacity(n):
    return 1 << max(0, int(n-1).bit_length())


def build_forward_inputs(rows, width, batch_capacity, pages, block_size):
    """CPU packing for one ancestor-masked forward, including chain rows.

    rows contain tokens=[recovery]+nodes, parent indices (root=-1), prefix
    length (excluding recovery), and a physical page table. Padding writes
    no KV and sees only a finite, real prefix. Logical slots and RoPE depth
    are intentionally independent.
    """
    cols = pages * block_size
    ids = np.zeros((batch_capacity,width),dtype=np.int64)
    rope = np.zeros_like(ids)
    slots = np.full((batch_capacity,width),-1,dtype=np.int32)
    indices = np.zeros((batch_capacity,pages),dtype=np.int32)
    mask = np.zeros((batch_capacity,width,cols),dtype=np.uint8)
    for b in range(batch_capacity):
        row = rows[b] if b < len(rows) else rows[0]
        prefix = int(row['prefix'])
        tokens, parents = row['tokens'], row['parents']
        n = len(tokens)
        if n != len(parents)+1 or n > width or prefix+n > cols:
            raise ValueError('Tree forward row exceeds its captured canvas')
        depth, _, visible = context_topology(parents)
        bt = row['blocks']
        live_pages = (prefix+n+block_size-1)//block_size
        if len(bt)<live_pages or any(p<0 for p in bt[:live_pages]):
            raise ValueError('Tree forward references an unallocated page')
        indices[b,:live_pages] = bt[:live_pages]
        indices[b,live_pages:] = bt[0]
        # Padding rows must not read an unwritten slot when prefix==0.
        mask[b,:,:max(prefix,1)] = 1
        if b >= len(rows):
            continue
        ids[b,:n] = tokens
        rope[b,:n] = prefix+np.asarray([0]+[d+1 for d in depth])
        positions = prefix+np.arange(n)
        slots[b,:n] = np.asarray(bt)[positions//block_size]*block_size+positions%block_size
        mask[b,:n,prefix:prefix+n] = visible
    packed = np.packbits(mask.reshape(-1),bitorder='little')
    return dict(ids=ids.reshape(-1),rope=rope.reshape(-1),
                slots=slots.reshape(-1),pages=indices.reshape(-1),mask=packed)


def restore_plan(previous, current_length, terminal, blocks, block_size):
    """Staged tree rows -> canonical slots after EOS/output truncation.

    A snapshot is looked up by sequence identity before calling this helper.
    Terminal identifies the untruncated verified path; length determines the
    committed prefix of that path, so later truncation cannot copy extra KV.
    """
    parents = previous['parents']
    accepted = int(current_length)-int(previous['length'])-1
    if accepted <= 0 or terminal <= 0:
        return [], []
    node = int(terminal)-1
    path = []
    while node >= 0:
        if node >= len(parents):
            raise ValueError('Terminal node is outside its sequence snapshot')
        path.append(node)
        node = parents[node]
    path.reverse()
    if accepted > len(path):
        raise ValueError('Committed tree length exceeds its terminal path')
    start = int(previous['length'])
    dst = [blocks[(start+j)//block_size]*block_size+(start+j)%block_size
           for j in range(accepted)]
    if any(s<0 for s in dst):
        raise ValueError('Accepted path references an unallocated page')
    return [j+1 for j in path[:accepted]], dst
