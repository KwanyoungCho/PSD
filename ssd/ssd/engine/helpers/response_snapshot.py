"""CPU view of one immutable fused response, shared by its consumers.

No model probability, token selection or wire layout changes. The caller
performs one bounded readback of the existing int64 response envelope.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HostResponse:
    cache_hits: tuple[int, ...]
    phase_source: tuple[int, ...]
    valid_k: tuple[int, ...]
    tokens: tuple[tuple[int, ...], ...]
    tree_ints: tuple[tuple[int, ...], ...] | None


def parse_host_response(values, batch, width, tree_extra=0):
    if batch < 1 or width < 1 or tree_extra < 0 or tree_extra % batch:
        raise ValueError('Invalid fused response dimensions')
    end=3*batch+batch*width
    if len(values)!=end+tree_extra:
        raise ValueError('Fused response length mismatch')
    def rows(start,columns):
        return tuple(tuple(int(v) for v in values[start+i*columns:start+(i+1)*columns]) for i in range(batch))
    valid=tuple(int(v) for v in values[2*batch:3*batch])
    if any(v<0 or v>width for v in valid):
        raise ValueError('Fused response valid_k outside token envelope')
    return HostResponse(tuple(int(v) for v in values[:batch]),
        tuple(int(v) for v in values[batch:2*batch]),valid,
        rows(3*batch,width),rows(end,tree_extra//batch) if tree_extra else None)
