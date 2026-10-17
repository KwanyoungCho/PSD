"""Immutable small topology packets prepared before the next cache request.

No vocabulary logits are materialized. Validation still happens for the hit
root before any response is sent. Python integers avoid repeated CPU-tensor
scalar conversions during that validation.
"""
from dataclasses import dataclass
import torch
from ssd.engine.helpers.p2_tree import tree_wire_ints_len, parse_tree_ints, validate_tree_ints


@dataclass(frozen=True)
class ResponseTopology:
    capacity: int
    device_rows: torch.Tensor
    host_rows: torch.Tensor
    integer_rows: tuple

    def select(self, root, vocab_size):
        if not 0 <= root < len(self.integer_rows):
            raise IndexError('Response root outside topology snapshot')
        parsed = parse_tree_ints(self.integer_rows[root], self.capacity)
        validate_tree_ints(parsed, self.capacity, vocab_size)
        return self.device_rows[root], self.host_rows[root], parsed


def prepare_response_topology(views, capacity):
    count, width = views['tok'].shape
    copy_n = min(width, capacity)
    packed = torch.zeros((count, tree_wire_ints_len(capacity)),
                         dtype=torch.int64, device=views['valid'].device)
    packed[:, 0].copy_(views['valid'])
    packed[:, 1].copy_(views['u_valid'])
    packed[:, 2].fill_(1)
    for i, name in enumerate(('tok', 'parent_local', 'sib_order', 'parent_q_ref')):
        packed[:, 3+i*capacity:3+i*capacity+copy_n].copy_(views[name][:, :copy_n])
    host = packed.detach().cpu()
    integers = tuple(tuple(row) for row in host.tolist())
    return ResponseTopology(capacity, packed, host, integers)
