"""AWQ-style W4A16 quantization integration for SSD (plan v2).

Backend: sgl-kernel's `gptq_marlin_gemm` with `uint4` ScalarType.
External AWQ checkpoints are repacked into Marlin layout via `awq_marlin_repack`.

Public surface:
    QuantConfig        — structured config (plan §13)
    AwqQuantState      — rank-local packed weight + scales + zeros + metadata
    awq_matmul         — Marlin matmul wrapper (runtime op)
    quant_init_context — context manager for meta-device linear __init__
    load_awq_artifact  — SSD-native artifact loader (Phase 4)
"""
from ssd.quant.config import QuantConfig
from ssd.quant.state import AwqQuantState
from ssd.quant.marlin import awq_matmul
from ssd.quant.init_context import quant_init_context, is_quant_init_active
from ssd.quant.io import (
    load_awq_artifact,
    save_awq_artifact,
    SSD_AWQ_ARTIFACT_VERSION,
)

__all__ = [
    "QuantConfig",
    "AwqQuantState",
    "awq_matmul",
    "quant_init_context",
    "is_quant_init_active",
    "load_awq_artifact",
    "save_awq_artifact",
    "SSD_AWQ_ARTIFACT_VERSION",
]
