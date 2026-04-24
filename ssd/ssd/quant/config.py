"""Structured QuantConfig — plan §13."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantConfig:
    enabled: bool = False
    method: str = "none"              # "none" | "awq_int4"
    target: bool = True
    draft: bool = False
    quantize_lm_head: bool = False
    quantize_embeddings: bool = False
    artifact_path: str | None = None
    artifact_mode: str = "load_only"  # "load_only" | "import_then_load"
    runtime_backend: str = "awq_marlin"   # only supported backend in this plan
    quant_source: str = "ssd_artifact"    # "ssd_artifact" | "external_awq"
    external_quant_path: str | None = None
    group_size: int = 128
    use_zero_point: bool = True
    # dtype of activations at runtime — Marlin accepts fp16 or bf16.
    expected_runtime_dtype: str = "float16"


def quant_config_from_legacy_flags(cfg) -> QuantConfig | None:
    """Build a QuantConfig from the legacy flat fields on ssd.config.Config.

    Plan §13.3 says keep the flat fields as a compat shim and derive the
    structured config at the LLM/runner boundary. This helper does the
    derivation; the AWQ runtime branch then reads the structured object,
    not the flat fields.

    Returns None when the AWQ-Marlin path is not active. The legacy
    torchao int4/int8 path does not use QuantConfig — it keeps reading
    the flat fields directly from `Config`.
    """
    if not getattr(cfg, "target_quant_enabled", False):
        return None
    backend = getattr(cfg, "target_quant_backend", "int4_wo_tile")
    if backend != "awq_marlin":
        # Legacy torchao path — QuantConfig unused, keep returning None so
        # callers don't accidentally run the AWQ branch with stale data.
        return None

    awq_artifact_path = getattr(cfg, "target_quant_awq_artifact", None)
    external_path = getattr(cfg, "target_quant_external_awq_path", None)
    source = "external_awq" if external_path else "ssd_artifact"

    import torch
    rt_dtype = cfg.hf_config.torch_dtype if cfg.hf_config is not None else torch.float16
    rt_dtype_str = str(rt_dtype).replace("torch.", "")

    return QuantConfig(
        enabled=True,
        method="awq_int4",
        target=True,
        draft=False,
        quantize_lm_head=getattr(cfg, "target_quant_lm_head", False),
        quantize_embeddings=False,
        artifact_path=awq_artifact_path,
        artifact_mode="load_only",
        runtime_backend="awq_marlin",
        quant_source=source,
        external_quant_path=external_path,
        group_size=getattr(cfg, "target_quant_group_size", 128),
        use_zero_point=True,
        expected_runtime_dtype=rt_dtype_str,
    )
