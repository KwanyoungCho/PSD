from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Int8WeightOnlyState:
    qweight: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None = None
    scheme: str = "per_channel_symmetric"


def quantize_weight_per_channel_int8(
    weight: torch.Tensor,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8 quantization.

    Args:
        weight: [out_features, in_features] float tensor.
        scale_dtype: dtype for stored scales.
    Returns:
        qweight: int8 tensor with same shape as weight.
        scales: [out_features] tensor.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape={tuple(weight.shape)}")

    weight_f = weight.float()
    max_abs = weight_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (max_abs / 127.0).squeeze(1).to(scale_dtype)
    qweight = torch.round(weight_f / scales.float().unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return qweight.contiguous(), scales.contiguous()


def dequantize_weight_per_channel_int8(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if qweight.dim() != 2:
        raise ValueError(f"Expected 2D qweight tensor, got shape={tuple(qweight.shape)}")
    if scales.dim() != 1 or scales.shape[0] != qweight.shape[0]:
        raise ValueError(
            f"Expected scales shape [{qweight.shape[0]}], got {tuple(scales.shape)}"
        )
    return qweight.to(out_dtype) * scales.to(out_dtype).unsqueeze(1)


def int8_weight_only_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correctness-first v1 implementation.

    This intentionally dequantizes on the fly before F.linear. It is not meant to be
    the final fast path; it is the safest path for bring-up and graph-contract validation.
    """
    weight = dequantize_weight_per_channel_int8(qweight, scales, x.dtype)
    return F.linear(x, weight, bias)
