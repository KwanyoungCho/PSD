"""Build an `AwqQuantState` from AutoAWQ-format tensors.

This is the bridge between:
  (a) Phase 3a: external AutoAWQ safetensors checkpoint → SSD TP module
  (b) Phase 3b: SSD-native artifact (also stored in AutoAWQ layout) → module
  (c) Phase 3b CPU importer: RTN-quantized dense weight → stored artifact

The steps done here:
  1. concat packed modules (qkv / gate_up) in AWQ layout (CPU)
  2. TP-shard along the correct axis (col = out, row = in + groups)
  3. move to CUDA
  4. `awq_marlin_repack` the qweight → Marlin layout
  5. Marlin-repack qzeros (reuse `awq_marlin_repack` with bits=4 on the zeros-as-qweight trick)
  6. build AwqQuantState

Shape contracts per `AwqQuantState.state.py` docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ssd.quant.state import AwqQuantState
from ssd.quant.marlin import repack_awq_to_marlin
from ssd.quant.marlin_utils import (
    marlin_permute_scales,
    marlin_zero_points_from_awq,
)


@dataclass
class RawAwqTensors:
    """AutoAWQ-layout trio for a single HF linear projection, CPU, full-rank."""
    qweight: torch.Tensor   # [in_features, out_features // 8] int32
    qzeros: torch.Tensor    # [num_groups,  out_features // 8] int32
    scales: torch.Tensor    # [num_groups,  out_features] fp16/bf16
    in_features: int
    out_features: int
    group_size: int
    bias: Optional[torch.Tensor] = None
    # For packed modules (qkv, gate_up), the full-rank out_features of each
    # sub-projection in concat order. When set, the TP-shard helper slices
    # each sub-projection independently and concatenates the per-rank
    # slices — matching SSD's dense QKVParallelLinear/MergedColumnParallel
    # loader semantics. When None, the whole tensor is sliced uniformly,
    # which is correct for standalone projections like o_proj/down_proj
    # and for packed modules whose parts happen to be equal-sized.
    part_out_features: Optional[list] = None


def concat_packed_awq(
    parts: Sequence[RawAwqTensors],
) -> RawAwqTensors:
    """Concatenate several AWQ-format projections along the output dim.

    Used for `qkv_proj` (q→k→v) and `gate_up_proj` (gate→up).

    Assumes identical in_features and group_size across parts; out_features
    are summed. Plan §9.3.1 concat order: caller must pass parts already in
    the correct order.
    """
    assert len(parts) >= 2
    ref = parts[0]
    for p in parts[1:]:
        assert p.in_features == ref.in_features, \
            f"concat_packed_awq: in_features mismatch {p.in_features} vs {ref.in_features}"
        assert p.group_size == ref.group_size
        assert p.scales.dtype == ref.scales.dtype

    total_out = sum(p.out_features for p in parts)
    qweight = torch.cat([p.qweight for p in parts], dim=1)   # pack is along dim=1
    qzeros = torch.cat([p.qzeros for p in parts], dim=1)
    scales = torch.cat([p.scales for p in parts], dim=1)
    bias = None
    if any(p.bias is not None for p in parts):
        bias = torch.cat([
            p.bias if p.bias is not None else torch.zeros(p.out_features, dtype=ref.scales.dtype)
            for p in parts
        ], dim=0)
    return RawAwqTensors(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        in_features=ref.in_features,
        out_features=total_out,
        group_size=ref.group_size,
        bias=bias,
        part_out_features=[p.out_features for p in parts],
    )


def shard_awq_column_parallel(
    raw: RawAwqTensors,
    tp_rank: int,
    tp_size: int,
) -> RawAwqTensors:
    """ColumnParallel shard (optionally sub-part-aware for packed QKV / gate_up).

    If `raw.part_out_features` is set, we slice each sub-projection
    independently by `part // tp_size` and concatenate the per-rank slices
    — matching the dense QKVParallelLinear / MergedColumnParallelLinear
    weight_loader convention. For Llama-3 GQA this is *required*: uniform
    slicing would split q (32 heads) and k/v (8 heads each) together into
    a single 1/tp_size slice, which mis-aligns the head boundaries.

    If `part_out_features` is None, slice uniformly.
    """
    pack = 8

    def _slice(t: torch.Tensor, start_col: int, width_col: int, dim: int) -> torch.Tensor:
        return t.narrow(dim, start_col, width_col).contiguous()

    if raw.part_out_features is None:
        # Uniform slice (single projection or equal-sized parts)
        parts = [raw.out_features]
    else:
        parts = list(raw.part_out_features)
        assert sum(parts) == raw.out_features

    qw_slices, qz_slices, sc_slices, bias_slices = [], [], [], []
    qw_cursor_packed = 0
    sc_cursor = 0
    for part_out in parts:
        assert part_out % tp_size == 0, \
            f"ColumnParallel AWQ shard: part_out={part_out} not /tp_size={tp_size}"
        per_rank_out = part_out // tp_size
        per_rank_packed = per_rank_out // pack

        packed_start = qw_cursor_packed + tp_rank * per_rank_packed
        sc_start = sc_cursor + tp_rank * per_rank_out

        qw_slices.append(_slice(raw.qweight, packed_start, per_rank_packed, dim=1))
        qz_slices.append(_slice(raw.qzeros, packed_start, per_rank_packed, dim=1))
        sc_slices.append(_slice(raw.scales, sc_start, per_rank_out, dim=1))
        if raw.bias is not None:
            bias_slices.append(_slice(raw.bias, sc_start, per_rank_out, dim=0))

        qw_cursor_packed += part_out // pack
        sc_cursor += part_out

    qweight = torch.cat(qw_slices, dim=1).contiguous()
    qzeros = torch.cat(qz_slices, dim=1).contiguous()
    scales = torch.cat(sc_slices, dim=1).contiguous()
    bias = torch.cat(bias_slices, dim=0).contiguous() if bias_slices else None

    total_out_local = sum(p // tp_size for p in parts)
    return RawAwqTensors(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        in_features=raw.in_features,
        out_features=total_out_local,
        group_size=raw.group_size,
        bias=bias,
        part_out_features=[p // tp_size for p in parts] if raw.part_out_features else None,
    )


def shard_awq_row_parallel(
    raw: RawAwqTensors,
    tp_rank: int,
    tp_size: int,
) -> RawAwqTensors:
    """RowParallel shard: in_features // tp_size (dim=1 of dense).

    For qweight (shape [in, out//8]), shard on dim=0 by `in_features // tp_size`.
    For qzeros (shape [num_groups, out//8]) and scales (shape [num_groups, out]),
    shard on dim=0 by `num_groups // tp_size`.

    Plan §9.3.4: assert shard_size % group_size == 0.
    """
    assert raw.in_features % tp_size == 0, \
        f"RowParallel AWQ shard: in_features={raw.in_features} not /tp_size={tp_size}"
    k_per_rank = raw.in_features // tp_size
    assert k_per_rank % raw.group_size == 0, \
        (f"RowParallel AWQ shard: shard_size={k_per_rank} not divisible by "
         f"group_size={raw.group_size} (plan §9.3.4)")
    groups_per_rank = k_per_rank // raw.group_size
    k_start = tp_rank * k_per_rank
    g_start = tp_rank * groups_per_rank

    qweight = raw.qweight[k_start : k_start + k_per_rank, :].contiguous()
    qzeros = raw.qzeros[g_start : g_start + groups_per_rank, :].contiguous()
    scales = raw.scales[g_start : g_start + groups_per_rank, :].contiguous()
    # RowParallel bias is shared across ranks (only rank 0 applies it in dense
    # path). Keep full bias on every rank and let the linear module handle it.
    return RawAwqTensors(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        in_features=k_per_rank,
        out_features=raw.out_features,
        group_size=raw.group_size,
        bias=raw.bias,
    )


def build_awq_state(raw: RawAwqTensors, device: torch.device) -> AwqQuantState:
    """Move AWQ tensors to `device`, Marlin-repack, and wrap in AwqQuantState.

    Three transforms done here (see `marlin_utils.py` for the math):
      - qweight : `awq_marlin_repack` (sgl_kernel CUDA op)
      - scales  : `marlin_permute_scales` (column perm inside 64-chunks)
      - qzeros  : AWQ-unpack → column perm → 8-interleave → int32-pack
                  via `marlin_zero_points_from_awq`

    The Phase 0 smoke test's ~0.20 rel-err was caused by skipping the
    scales/qzeros permutations; with them the error collapses to dense
    matmul roundoff.
    """
    qw_cuda = raw.qweight.to(device, non_blocking=True).contiguous()
    qz_cuda = raw.qzeros.to(device, non_blocking=True).contiguous()
    sc_cuda = raw.scales.to(device, non_blocking=True).contiguous()

    marlin_qweight = repack_awq_to_marlin(
        qw_cuda, size_k=raw.in_features, size_n=raw.out_features, bits=4,
    )
    marlin_scales = marlin_permute_scales(
        sc_cuda, size_k=raw.in_features, size_n=raw.out_features,
        group_size=raw.group_size,
    )
    marlin_qzeros = marlin_zero_points_from_awq(
        qz_cuda, size_k=raw.in_features, size_n=raw.out_features, num_bits=4,
    )

    bias = raw.bias.to(device, non_blocking=True).contiguous() if raw.bias is not None else None
    workspace = AwqQuantState.marlin_workspace(raw.out_features, device)

    return AwqQuantState(
        marlin_qweight=marlin_qweight,
        scales=marlin_scales,
        qzeros=marlin_qzeros,
        workspace=workspace,
        bias=bias,
        in_features=raw.in_features,
        out_features=raw.out_features,
        group_size=raw.group_size,
        bits=4,
        scalar_type_name="uint4",
    )
