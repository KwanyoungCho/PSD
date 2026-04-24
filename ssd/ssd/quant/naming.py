"""Llama-family HF name → SSD module name mapping.

Plan §9.2 requirement. Hard-coded for Llama-family because that's the
Phase-1 scope (§10). Extensible later for Qwen3 by adding a per-family
mapping dict.

HF checkpoint keys (AutoAWQ stores the same prefix with .qweight/.qzeros/.scales):
    model.layers.{i}.self_attn.q_proj       → SSD qkv_proj, shard "q"
    model.layers.{i}.self_attn.k_proj       → SSD qkv_proj, shard "k"
    model.layers.{i}.self_attn.v_proj       → SSD qkv_proj, shard "v"
    model.layers.{i}.self_attn.o_proj       → SSD o_proj     (row-parallel)
    model.layers.{i}.mlp.gate_proj          → SSD gate_up_proj, shard 0
    model.layers.{i}.mlp.up_proj            → SSD gate_up_proj, shard 1
    model.layers.{i}.mlp.down_proj          → SSD down_proj   (row-parallel)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShardMode(Enum):
    COLUMN = "column"     # ColumnParallel — shard out_features
    ROW = "row"           # RowParallel — shard in_features + groups


@dataclass(frozen=True)
class SsdModuleSpec:
    ssd_name: str
    shard_mode: ShardMode
    # For packed modules, the parts in concat order. None for single modules.
    parts: tuple[str, ...] | None = None
    # When concatenating QKV, we pass "q"/"k"/"v" shard ids; here we don't care.


# HF linear module suffix → how SSD handles it.
# Keys are the "local" HF name (final component). Used for name mapping below.
_LLAMA_LINEAR_MAP = {
    # attention
    "self_attn.q_proj": ("self_attn.qkv_proj", ShardMode.COLUMN, "q"),
    "self_attn.k_proj": ("self_attn.qkv_proj", ShardMode.COLUMN, "k"),
    "self_attn.v_proj": ("self_attn.qkv_proj", ShardMode.COLUMN, "v"),
    "self_attn.o_proj": ("self_attn.o_proj", ShardMode.ROW, None),
    # mlp
    "mlp.gate_proj": ("mlp.gate_up_proj", ShardMode.COLUMN, "gate"),
    "mlp.up_proj": ("mlp.gate_up_proj", ShardMode.COLUMN, "up"),
    "mlp.down_proj": ("mlp.down_proj", ShardMode.ROW, None),
}


def llama_layer_ssd_modules(num_layers: int) -> list[SsdModuleSpec]:
    """Return the SSD module list (column + row parallel linears) for a Llama
    model with `num_layers` decoder layers. Does not include lm_head / embeddings.
    """
    modules: list[SsdModuleSpec] = []
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        modules.append(SsdModuleSpec(
            ssd_name=f"{prefix}.self_attn.qkv_proj",
            shard_mode=ShardMode.COLUMN,
            parts=("q", "k", "v"),
        ))
        modules.append(SsdModuleSpec(
            ssd_name=f"{prefix}.self_attn.o_proj",
            shard_mode=ShardMode.ROW,
        ))
        modules.append(SsdModuleSpec(
            ssd_name=f"{prefix}.mlp.gate_up_proj",
            shard_mode=ShardMode.COLUMN,
            parts=("gate", "up"),
        ))
        modules.append(SsdModuleSpec(
            ssd_name=f"{prefix}.mlp.down_proj",
            shard_mode=ShardMode.ROW,
        ))
    return modules


def hf_linear_to_ssd(hf_name: str) -> tuple[str, ShardMode, str | None] | None:
    """Map one HF linear module name (without .weight/.qweight suffix) to its
    SSD module + shard id, or None if not a quantizable linear.

    Example: "model.layers.3.self_attn.q_proj"
             → ("model.layers.3.self_attn.qkv_proj", COLUMN, "q")
    """
    for suffix, (target_suffix, mode, shard_id) in _LLAMA_LINEAR_MAP.items():
        if hf_name.endswith("." + suffix):
            prefix = hf_name[: -len(suffix)]
            return prefix + target_suffix, mode, shard_id
    return None
