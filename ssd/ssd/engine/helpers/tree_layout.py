"""TreeLayout: abstracts tree decode layout so _decode_tree and CudaGraph helpers
are parametrized by layout instead of global MQ_LEN. Supports full / draft / proxy /
phase1 layouts.

For Phase 2 hybrid (per MESA-PHASE2-HYBRID-IMPLEMENTATION-PLAN.md):

- ``forward_depth`` (alias: ``K``) — how many model-forward steps a seed runs
  through. ``arange(forward_depth)`` drives ``step_pos_offsets`` /
  ``step_rope_offsets``.
- ``position_count`` (= ``len(fan_out_list)``) — how many fork positions the
  layout has. Glue-position offset uses this.

For non-MESA layouts, the legacy invariant ``position_count == forward_depth + 1``
continues to hold and nothing visible changes. For MESA Phase 1 layouts the two
diverge: ``forward_depth = K1`` (config-fixed), ``position_count = valid_k + 1``
(variable per step / per bucket).
"""

from dataclasses import dataclass
import torch


@dataclass
class TreeLayout:
    name: str                          # "full", "draft", "proxy", "phase1_long", ...
    fan_out_list: list[int]            # per-position fan_out [position_count]
    fan_out_list_miss: list[int]       # per-position fan_out for cache miss [position_count]
    MQ_LEN: int                        # sum(fan_out_list)
    K: int                             # forward depth (a.k.a. forward_depth)
    position_count: int                # len(fan_out_list); decoupled from K

    # Pre-computed tensors (on device)
    fan_out_t: torch.Tensor            # tensor(fan_out_list)
    fan_out_t_miss: torch.Tensor       # tensor(fan_out_list_miss)
    fan_idx_hit: torch.Tensor          # arange(position_count).repeat_interleave(fan_out_t)
    fan_idx_miss: torch.Tensor         # arange(position_count).repeat_interleave(fan_out_t_miss)
    arange_mq: torch.Tensor            # arange(MQ_LEN)
    step_pos_offsets: torch.Tensor     # arange(forward_depth)[:, None] * MQ_LEN
    step_rope_offsets: torch.Tensor    # arange(forward_depth)[:, None]

    graph_key: str                     # CudaGraph / graph_vars lookup key

    @property
    def forward_depth(self) -> int:
        """Alias for ``K`` — the number of model-forward steps in the depth loop."""
        return self.K


def create_tree_layout(
    name: str,
    fan_out_list: list[int],
    fan_out_list_miss: list[int],
    K: int,
    device: torch.device,
    position_count: int | None = None,
) -> TreeLayout:
    """Create a TreeLayout.

    Args:
        name: layout name (used as graph_key suffix).
        fan_out_list: per-position fan_out (length = position_count).
        fan_out_list_miss: per-position fan_out for cache miss case.
        K: forward depth (number of decode iterations of the depth loop).
        device: target device for pre-computed tensors.
        position_count: number of fork positions (= ``len(fan_out_list)``). When
            None (default), defaults to ``K + 1`` to preserve the legacy
            invariant for non-MESA / current MESA call sites.
    """
    MQ_LEN = sum(fan_out_list)
    assert sum(fan_out_list_miss) == MQ_LEN, \
        f"fan_out_list_miss sum ({sum(fan_out_list_miss)}) must equal fan_out_list sum ({MQ_LEN})"

    if position_count is None:
        position_count = K + 1
    assert len(fan_out_list) == position_count, \
        f"len(fan_out_list)={len(fan_out_list)} must equal position_count={position_count}"
    assert len(fan_out_list_miss) == position_count, \
        f"len(fan_out_list_miss)={len(fan_out_list_miss)} must equal position_count={position_count}"

    fan_out_t = torch.tensor(fan_out_list, device=device, dtype=torch.int64)
    fan_out_t_miss = torch.tensor(fan_out_list_miss, device=device, dtype=torch.int64)

    return TreeLayout(
        name=name,
        fan_out_list=fan_out_list,
        fan_out_list_miss=fan_out_list_miss,
        MQ_LEN=MQ_LEN,
        K=K,
        position_count=position_count,
        fan_out_t=fan_out_t,
        fan_out_t_miss=fan_out_t_miss,
        fan_idx_hit=torch.arange(position_count, device=device, dtype=torch.int64).repeat_interleave(fan_out_t),
        fan_idx_miss=torch.arange(position_count, device=device, dtype=torch.int64).repeat_interleave(fan_out_t_miss),
        arange_mq=torch.arange(MQ_LEN, device=device, dtype=torch.int64),
        step_pos_offsets=torch.arange(K, device=device, dtype=torch.int64)[:, None] * MQ_LEN,
        step_rope_offsets=torch.arange(K, device=device, dtype=torch.int64)[:, None],
        graph_key=f"fi_tree_decode_{name}" if name != "full" else "fi_tree_decode",
    )
