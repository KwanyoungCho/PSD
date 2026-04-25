"""Marlin scale/zero-point permutation helpers.

Ported from vLLM's `model_executor/layers/quantization/utils/marlin_utils.py`
(Apache-2.0). `gptq_marlin_gemm` expects scales and qzeros in a specific
permuted layout that `awq_marlin_repack` does NOT produce — it only handles
the packed weight. For AWQ input tensors we must also:

  - permute scales columns (`marlin_permute_scales`)
  - unpack AWQ qzeros, permute, and repack into Marlin's int32 layout
    (`marlin_zero_points_from_awq`)

All three outputs are what `gptq_marlin_gemm(b_q_type=uint4, is_zp_float=False)`
consumes.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ssd.quant.pack import awq_unpack_4bit


def get_scale_perms() -> Tuple[list, list]:
    scale_perm: list = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]]
        )
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    """Permute scales into Marlin layout.

    Input shape:  [num_groups, size_n]      (AutoAWQ-native, same as scales
                                             field in our AwqQuantState)
    Output shape: [num_groups, size_n]      (same shape, permuted columns
                                             inside every 64-wide chunk)
    """
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        perm = scale_perm
    else:
        perm = scale_perm_single
    perm_t = torch.tensor(perm, device=scales.device, dtype=torch.long)
    s = scales.reshape(-1, len(perm)).index_select(1, perm_t)
    s = s.reshape(-1, size_n).contiguous()
    return s


def _pack_cols_int32(
    q_w: torch.Tensor,
    *,
    num_bits: int,
) -> torch.Tensor:
    """Pack small-int values along the last dim into int32 (plain bit-shift)."""
    assert q_w.dim() == 2
    pack_factor = 32 // num_bits
    rows, cols = q_w.shape
    assert cols % pack_factor == 0, f"pack_cols: cols={cols} not /{pack_factor}"
    q_w = q_w.to(torch.int32) & ((1 << num_bits) - 1)
    shifts = torch.arange(pack_factor, device=q_w.device, dtype=torch.int32) * num_bits
    # [rows, cols] → [rows, cols // pack_factor, pack_factor] → bit-OR
    reshaped = q_w.view(rows, cols // pack_factor, pack_factor)
    packed = (reshaped * (1 << shifts).to(torch.int32)).sum(dim=-1).to(torch.int32)
    return packed.contiguous()


def marlin_zero_points_from_awq(
    awq_qzeros: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Convert AutoAWQ qzeros → Marlin qzeros.

    Input:  awq_qzeros [num_groups, size_n // 8] int32  (AutoAWQ-packed)
    Output: marlin_qz  [num_groups, size_n // 8] int32  (Marlin-packed)

    Steps (matches vLLM `unpack_awq_zeros` + `marlin_zero_points`):
        1. AWQ-unpack qzeros → [num_groups, size_n] int32
        2. apply `scale_perm` to columns inside every 64-wide chunk
        3. apply 8-wide interleave [0,2,4,6,1,3,5,7] within every 8-chunk
        4. plain int32 pack along cols (8 int4 per int32, no interleave)
    """
    # Step 1: AWQ unpack — layout is 8 packed values per int32 with
    # interleave [0, 2, 4, 6, 1, 3, 5, 7]; awq_unpack_4bit handles the
    # inverse, producing [num_groups, size_n] int32 with values in [0, 15].
    z = awq_unpack_4bit(awq_qzeros, pack_dim=-1)    # [num_groups, size_n]
    assert z.shape[1] == size_n, (z.shape, size_n)

    # Step 2: scale_perm on columns
    scale_perm, scale_perm_single = get_scale_perms()
    perm_t = torch.tensor(scale_perm, device=z.device, dtype=torch.long)
    z = z.reshape(-1, len(scale_perm)).index_select(1, perm_t)
    # shape: [num_groups * (size_n // 64), 64]

    # Step 3: 8-wide interleave
    if num_bits == 4:
        interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=z.device, dtype=torch.long)
    else:
        assert num_bits == 8
        interleave = torch.tensor([0, 2, 1, 3], device=z.device, dtype=torch.long)
    z = z.reshape(-1, len(interleave)).index_select(1, interleave).reshape(-1)
    z = z.reshape(-1, size_n).contiguous()

    # Step 4: plain int32 pack along cols
    return _pack_cols_int32(z, num_bits=num_bits)
