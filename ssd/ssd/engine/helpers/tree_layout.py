"""TreeLayout: abstracts tree decode layout so _decode_tree and CudaGraph helpers
are parametrized by layout instead of global MQ_LEN.  Supports full / draft / proxy layouts."""

from dataclasses import dataclass
import torch


@dataclass
class TreeLayout:
    name: str                          # "full", "draft", "proxy"
    fan_out_list: list[int]            # per-position fan_out [K+1]
    fan_out_list_miss: list[int]       # per-position fan_out for cache miss [K+1]
    MQ_LEN: int                        # sum(fan_out_list)
    K: int                             # speculate_k

    # Pre-computed tensors (on device)
    fan_out_t: torch.Tensor            # tensor(fan_out_list)
    fan_out_t_miss: torch.Tensor       # tensor(fan_out_list_miss)
    fan_idx_hit: torch.Tensor          # arange(K+1).repeat_interleave(fan_out_t)
    fan_idx_miss: torch.Tensor         # arange(K+1).repeat_interleave(fan_out_t_miss)
    arange_mq: torch.Tensor            # arange(MQ_LEN)
    step_pos_offsets: torch.Tensor     # arange(K)[:, None] * MQ_LEN
    step_rope_offsets: torch.Tensor    # arange(K)[:, None]

    graph_key: str                     # CudaGraph / graph_vars lookup key


def create_tree_layout(
    name: str,
    fan_out_list: list[int],
    fan_out_list_miss: list[int],
    K: int,
    device: torch.device,
) -> TreeLayout:
    MQ_LEN = sum(fan_out_list)
    assert sum(fan_out_list_miss) == MQ_LEN, \
        f"fan_out_list_miss sum ({sum(fan_out_list_miss)}) must equal fan_out_list sum ({MQ_LEN})"
    assert len(fan_out_list) == K + 1
    assert len(fan_out_list_miss) == K + 1

    fan_out_t = torch.tensor(fan_out_list, device=device, dtype=torch.int64)
    fan_out_t_miss = torch.tensor(fan_out_list_miss, device=device, dtype=torch.int64)

    return TreeLayout(
        name=name,
        fan_out_list=fan_out_list,
        fan_out_list_miss=fan_out_list_miss,
        MQ_LEN=MQ_LEN,
        K=K,
        fan_out_t=fan_out_t,
        fan_out_t_miss=fan_out_t_miss,
        fan_idx_hit=torch.arange(K + 1, device=device, dtype=torch.int64).repeat_interleave(fan_out_t),
        fan_idx_miss=torch.arange(K + 1, device=device, dtype=torch.int64).repeat_interleave(fan_out_t_miss),
        arange_mq=torch.arange(MQ_LEN, device=device, dtype=torch.int64),
        step_pos_offsets=torch.arange(K, device=device, dtype=torch.int64)[:, None] * MQ_LEN,
        step_rope_offsets=torch.arange(K, device=device, dtype=torch.int64)[:, None],
        graph_key=f"fi_tree_decode_{name}" if name != "full" else "fi_tree_decode",
    )
