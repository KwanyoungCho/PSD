"""Structured QuantConfig — plan §13, role-aware (target / draft).

One dataclass, role-aware. A single `Config` may yield up to two
`QuantConfig` instances — one for target (`role="target"`) and one for
draft (`role="draft"`). They are independent: either can be enabled or
disabled without constraining the other.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Role-aware (target/draft) AWQ runtime config consumed by ModelRunner."""

    role: str                              # "target" | "draft"
    enabled: bool = False
    artifact_path: str | None = None       # SSD-native artifact prefix
    quant_source: str = "ssd_artifact"     # "ssd_artifact" | "external_awq"
    external_quant_path: str | None = None
    group_size: int = 128
    expected_runtime_dtype: str = "float16"


def _dtype_str(cfg, is_draft: bool) -> str:
    import torch
    hf = cfg.draft_hf_config if is_draft else cfg.hf_config
    if hf is None:
        return "float16"
    dt = hf.torch_dtype
    return str(dt).replace("torch.", "")


def quant_config_from_legacy_flags(cfg, role: str) -> QuantConfig | None:
    """Derive a role-specific `QuantConfig` from flat `{role}_quant_*` flags.

    Returns None when AWQ is disabled for that role, or when the role is
    `"target"` and a legacy torchao backend is selected (the runner's
    deprecated `[LEGACY]` branch consumes the flat fields directly in that
    case — this function does not produce a QuantConfig for it).
    """
    assert role in ("target", "draft"), f"role must be target|draft, got {role!r}"

    if role == "target":
        enabled = getattr(cfg, "target_quant_enabled", False)
        backend = getattr(cfg, "target_quant_backend", "awq_marlin")
        artifact = getattr(cfg, "target_quant_awq_artifact", None)
        external = getattr(cfg, "target_quant_external_awq_path", None)
        group_size = getattr(cfg, "target_quant_group_size", 128)
    else:  # draft
        enabled = getattr(cfg, "draft_quant_enabled", False)
        backend = getattr(cfg, "draft_quant_backend", "awq_marlin")
        artifact = getattr(cfg, "draft_quant_awq_artifact", None)
        external = getattr(cfg, "draft_quant_external_awq_path", None)
        group_size = getattr(cfg, "draft_quant_group_size", 128)

    # Auto-route: if the user passed an AWQ artifact/external path but left
    # `backend` at a stale legacy value (or omitted it), resolve to AWQ.
    if (artifact or external) and backend != "awq_marlin":
        print(
            f"[quant][{role}] auto-routing backend {backend!r} → 'awq_marlin' "
            f"because an AWQ artifact/external path was provided.",
            file=sys.stderr, flush=True,
        )
        backend = "awq_marlin"

    if not enabled:
        return None
    if backend != "awq_marlin":
        # Legacy torchao backend selected on the target. The deprecated
        # branch in `model_runner.py` reads the flat fields directly and
        # logs a `[LEGACY]` warning; we don't return a QuantConfig so the
        # AWQ runtime path stays inert.
        if role == "draft":
            raise ValueError(
                f"draft_quant_backend={backend!r} is not supported; "
                f"only 'awq_marlin' is supported for the draft."
            )
        return None

    is_draft = role == "draft"
    return QuantConfig(
        role=role,
        enabled=True,
        artifact_path=artifact,
        quant_source="external_awq" if external else "ssd_artifact",
        external_quant_path=external,
        group_size=group_size,
        expected_runtime_dtype=_dtype_str(cfg, is_draft),
    )
