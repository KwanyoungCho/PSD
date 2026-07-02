"""AWQ-format packing utilities — pure PyTorch, no kernel calls.

AutoAWQ stores a quantized linear as three tensors:
    qweight : int32 [in_features, out_features // pack_factor]
    qzeros  : int32 [in_features // group_size, out_features // pack_factor]
    scales  : dtype [in_features // group_size, out_features]

Within each int32 column 8 int4 values are packed with the AutoAWQ-specific
interleave order [0, 2, 4, 6, 1, 3, 5, 7]. Both qweight (per-row quant code)
and qzeros (per-group zero-point code) use the same packing convention.

`awq_marlin_repack` in sgl-kernel consumes this exact layout, so our job is
to make sure we both produce and ingest it correctly.
"""
from __future__ import annotations

from typing import Tuple

import torch


AWQ_PACK_FACTOR = 8   # W4: 32 / 4
AWQ_REVERSE_ORDER = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int32)


def awq_pack_4bit(values_int4: torch.Tensor, out_dim: int = 1) -> torch.Tensor:
    """Pack a [rows, N] int4 tensor along `out_dim` into AutoAWQ int32 layout.

    values_int4: integer tensor with values in [0, 15]. Must have N % 8 == 0.
    out_dim: which dim to pack along. 1 for qweight ([in, out]), 1 for qzeros.

    Returns int32 tensor with the packed dim shrunk by 8×.
    """
    assert values_int4.dtype in (torch.int32, torch.int64, torch.uint8), \
        f"awq_pack_4bit expects integer tensor, got {values_int4.dtype}"
    values = values_int4.to(torch.int32) & 0xF
    if out_dim != values.dim() - 1:
        # move packed dim to the last position
        values = values.transpose(out_dim, -1).contiguous()
    *lead, N = values.shape
    assert N % AWQ_PACK_FACTOR == 0, f"awq_pack_4bit: last dim {N} not /8"
    num_cols = N // AWQ_PACK_FACTOR

    order = AWQ_REVERSE_ORDER.to(values.device)
    packed = torch.zeros(*lead, num_cols, dtype=torch.int32, device=values.device)
    # For each output column we take 8 source ints and shift-or them.
    # Reshape once, gather in interleave order, then packed shift.
    reshaped = values.view(*lead, num_cols, AWQ_PACK_FACTOR)
    gathered = reshaped.index_select(-1, order)      # [..., num_cols, 8]
    shifts = (torch.arange(AWQ_PACK_FACTOR, device=values.device, dtype=torch.int32) * 4)
    packed = (gathered * (1 << shifts).to(torch.int32)).sum(dim=-1).to(torch.int32)

    if out_dim != values_int4.dim() - 1:
        # undo transpose
        packed = packed.transpose(out_dim, -1).contiguous()
    return packed


def awq_unpack_4bit(qweight: torch.Tensor, pack_dim: int = -1) -> torch.Tensor:
    """Inverse of `awq_pack_4bit`. Returns int32 with values in [0, 15]."""
    packed = qweight.to(torch.int32)
    if pack_dim != packed.dim() - 1:
        packed = packed.transpose(pack_dim, -1).contiguous()
    *lead, num_cols = packed.shape
    order = AWQ_REVERSE_ORDER.to(packed.device)
    shifts = (torch.arange(AWQ_PACK_FACTOR, device=packed.device, dtype=torch.int32) * 4)
    # expand: [..., num_cols, 8]
    extracted = (packed.unsqueeze(-1) >> shifts) & 0xF
    # invert the interleave: extracted[..., i] corresponds to original index order[i]
    out = torch.empty(*lead, num_cols, AWQ_PACK_FACTOR, dtype=torch.int32, device=packed.device)
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(AWQ_PACK_FACTOR, device=order.device, dtype=order.dtype)
    out = extracted.index_select(-1, inv_order)
    out = out.reshape(*lead, num_cols * AWQ_PACK_FACTOR)
    if pack_dim != qweight.dim() - 1:
        out = out.transpose(pack_dim, -1).contiguous()
    return out


def rtn_quantize_w4a16(
    weight: torch.Tensor,
    group_size: int = 128,
    use_zero_point: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-to-nearest W4A16 quantization — per-group along input dim.

    Used by the offline importer as a "no external tool needed" path, and
    by the DUET accept-rate comparison per plan §16.2 mitigation.

    Args:
        weight: dense fp16/bf16 weight [out_features, in_features]
        group_size: 128 (AutoAWQ / Marlin standard)
        use_zero_point: AWQ stores a per-group zero-point (uint4). Marlin
            consumes it via `b_zeros`.

    Returns:
        (qweight_awq, qzeros_awq, scales, dequant_weight)

        qweight_awq : int32 [in_features, out_features // 8]   AutoAWQ layout
        qzeros_awq  : int32 [num_groups, out_features // 8]    AutoAWQ layout
        scales      : dtype [num_groups, out_features]         row-major groups
        dequant_weight: dtype [out_features, in_features]      for verification
    """
    assert weight.dim() == 2, weight.shape
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, \
        f"RTN W4A16: in_features={in_features} not divisible by group_size={group_size}"
    assert out_features % AWQ_PACK_FACTOR == 0, \
        f"RTN W4A16: out_features={out_features} not divisible by 8"

    dtype = weight.dtype
    num_groups = in_features // group_size
    # Per-group min/max along in dim → asymmetric range [qmin, qmax] = [0, 15]
    w_grouped = weight.view(out_features, num_groups, group_size).to(torch.float32)

    if use_zero_point:
        w_min = w_grouped.amin(dim=-1)       # [out_features, num_groups]
        w_max = w_grouped.amax(dim=-1)
        scale_per_group = ((w_max - w_min) / 15.0).clamp_min(1e-8)
        zero_per_group = (-w_min / scale_per_group).round().clamp(0, 15).to(torch.int32)
        # Quantize
        q = (w_grouped / scale_per_group.unsqueeze(-1) + zero_per_group.unsqueeze(-1)).round()
        q = q.clamp(0, 15).to(torch.int32)   # [out_features, num_groups, group_size]
    else:
        w_absmax = w_grouped.abs().amax(dim=-1).clamp_min(1e-8)
        scale_per_group = (w_absmax / 7.0)
        zero_per_group = torch.full_like(scale_per_group, 8, dtype=torch.int32)
        q = (w_grouped / scale_per_group.unsqueeze(-1) + 8).round()
        q = q.clamp(0, 15).to(torch.int32)

    # Dequantize for verification
    dequant = ((q - zero_per_group.unsqueeze(-1).to(torch.float32))
               * scale_per_group.unsqueeze(-1)).reshape(out_features, in_features).to(dtype)

    # Lay out for AWQ: qweight has shape [in_features, out_features // 8]
    # and qzeros [num_groups, out_features // 8]. Both pack 8 int4 values
    # along the *output* direction with the interleave order.
    q_in_by_out = q.view(out_features, in_features).T.contiguous()   # [in_features, out_features]
    qweight_awq = awq_pack_4bit(q_in_by_out, out_dim=1)              # [in_features, out//8]

    zeros_in_by_out = zero_per_group.T.contiguous()                  # [num_groups, out_features]
    qzeros_awq = awq_pack_4bit(zeros_in_by_out, out_dim=1)           # [num_groups, out//8]

    scales = scale_per_group.T.contiguous().to(dtype)                # [num_groups, out_features]
    return qweight_awq, qzeros_awq, scales, dequant
