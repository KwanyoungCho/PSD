"""Marlin W4A16 matmul wrapper — the only runtime op for the AWQ backend.

Uses sgl-kernel's `gptq_marlin_gemm`. The kernel handles fp16 or bf16
activations natively and is graph-safe (confirmed in Phase 0 smoke test).

External dependencies (installed in the `ssd` env):
  - sgl-kernel==0.3.17.post1  → gptq_marlin_gemm, awq_marlin_repack
"""
from __future__ import annotations

from typing import Optional

import torch

# Import lazily at call time would be safer against "sgl_kernel not installed"
# envs, but SSD already hard-depends on sgl-kernel via pyproject.toml so we
# keep the import at module load.
from sgl_kernel import gptq_marlin_gemm, awq_marlin_repack
from sgl_kernel.scalar_type import scalar_types

from ssd.quant.state import AwqQuantState


_SCALAR_BY_NAME = {
    "uint4": scalar_types.uint4,
    "uint4b8": scalar_types.uint4b8,
}


def awq_matmul(
    x: torch.Tensor,
    state: AwqQuantState,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward op for a TP linear module running AWQ W4A16 via Marlin.

    Args:
        x: [*, in_features] activation in fp16 or bf16
        state: AwqQuantState attached to the module
        bias: optional [out_features] bias (added post-matmul). Marlin's
              own bias arg is not used here — we add after to match the
              dense F.linear convention exactly.

    Returns:
        [*, out_features] tensor in `x.dtype`.
    """
    assert x.is_cuda, "awq_matmul: activation must be on CUDA"
    assert x.dtype in (torch.float16, torch.bfloat16), \
        f"awq_matmul: activation dtype must be fp16/bf16, got {x.dtype}"

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x2.shape
    N = state.out_features
    assert K == state.in_features, \
        f"awq_matmul: K mismatch x={K} vs state={state.in_features}"

    scalar_type = _SCALAR_BY_NAME[state.scalar_type_name]

    y = gptq_marlin_gemm(
        a=x2,
        c=None,
        b_q_weight=state.marlin_qweight,
        b_scales=state.scales,
        global_scale=None,
        b_zeros=state.qzeros,
        g_idx=None,
        perm=None,
        workspace=state.workspace,
        b_q_type=scalar_type,
        size_m=M,
        size_n=N,
        size_k=K,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    if bias is not None:
        y = y + bias
    return y.reshape(*orig_shape[:-1], N)


def repack_awq_to_marlin(
    awq_qweight: torch.Tensor,
    size_k: int,
    size_n: int,
    bits: int = 4,
) -> torch.Tensor:
    """Convert AutoAWQ-format qweight → Marlin-format qweight.

    AutoAWQ qweight shape: [size_k, size_n // pack_factor] int32,
    with 8 int4 values packed per int32 along the output dim with
    interleave order [0, 2, 4, 6, 1, 3, 5, 7].

    Marlin consumes a repacked layout returned by awq_marlin_repack.
    """
    return awq_marlin_repack(awq_qweight, size_k=size_k, size_n=size_n, num_bits=bits)
