"""AwqQuantState — the in-memory representation attached to a TP linear module.

The state is what the forward path sees. All tensors are rank-local and live
on the same CUDA device as the module. The state is NOT an `nn.Parameter`;
the TP linear module holds it as a regular attribute.

Shape contract (per TP rank, matches Marlin's expectations):
    in_features  = K_local   (= input_size for ColumnParallel, else sharded)
    out_features = N_local   (= output_size for RowParallel, else sharded)
    num_groups   = K_local // group_size
    pack_factor  = 32 // bits = 8 for W4

    marlin_qweight : int32 [K_local // 16, N_local * 2]   (post awq_marlin_repack)
    scales         : dtype [num_groups, N_local]          (row-major, groups along K)
    qzeros         : int32 [num_groups, N_local // pack_factor]
    workspace      : int32 [N_local // 64 * 16]           (scratch buffer for Marlin)
    bias           : dtype [N_local] or None

Notes:
  - `bits` is always 4 for this backend.
  - `scalar_type` stored as a string ("uint4" or "uint4b8") so the state is
    picklable; resolved to the sgl-kernel ScalarType at forward time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class AwqQuantState:
    marlin_qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor                 # Marlin-repacked zeros (not raw AWQ)
    workspace: torch.Tensor
    bias: Optional[torch.Tensor]

    in_features: int
    out_features: int
    group_size: int = 128
    bits: int = 4
    scalar_type_name: str = "uint4"      # AutoAWQ uses uint4; GPTQ uses uint4b8

    def device(self) -> torch.device:
        return self.marlin_qweight.device

    def dtype(self) -> torch.dtype:
        return self.scales.dtype

    def storage_bytes(self) -> int:
        """Best-effort sum of all storage held by this state."""
        seen = set()
        total = 0
        for t in (self.marlin_qweight, self.scales, self.qzeros, self.workspace, self.bias):
            if t is None:
                continue
            s = t.untyped_storage()
            key = (s.data_ptr(), s.size())
            if key in seen:
                continue
            seen.add(key)
            total += s.size()
        return total

    @staticmethod
    def marlin_workspace(out_features: int, device: torch.device) -> torch.Tensor:
        """Scratch buffer Marlin needs. 64 is the output tile size."""
        return torch.zeros(out_features // 64 * 16, dtype=torch.int32, device=device)
