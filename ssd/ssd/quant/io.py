"""SSD-native AWQ artifact format — plan §7.3 / §7.4.

Each per-rank file contains packed AWQ tensors (AutoAWQ layout, not yet
Marlin-repacked — we repack on load because the repack is a cheap CUDA op
and the Marlin layout is version-tied).

File layout (torch.save, pickle):

    {
        "schema_version": SSD_AWQ_ARTIFACT_VERSION,
        "quant_scheme": "awq_int4",
        "backend": "awq_marlin",
        "model_id": str,
        "tp_size": int,
        "tp_rank": int,
        "group_size": 128,
        "use_zero_point": True,
        "expected_runtime_dtype": "float16" | "bfloat16",
        "quantize_lm_head": bool,
        "quantize_embeddings": bool,
        "quant_source": "rtn" | "awq_calibrated" | "external",
        "ssd_module_names": list[str],       # `named_modules` path in SSD model
        "modules": {
            module_name: {
                "qweight":   int32  [K_local, N_local // 8],      AutoAWQ layout
                "qzeros":    int32  [K_local // G, N_local // 8], AutoAWQ layout
                "scales":    dtype  [K_local // G, N_local],
                "in_features":  int,
                "out_features": int,
                "group_size":   int,
                "bias":  dtype [N_local] or None,
            }, ...
        },
    }
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Mapping

import torch

SSD_AWQ_ARTIFACT_VERSION = 1


def _artifact_path(prefix: str, tp_rank: int) -> str:
    return f"{prefix}.rank{tp_rank}.awq.pt"


def save_awq_artifact(
    *,
    prefix: str,
    tp_rank: int,
    tp_size: int,
    modules: Mapping[str, Mapping[str, Any]],
    model_id: str,
    group_size: int,
    use_zero_point: bool,
    expected_runtime_dtype: str,
    quantize_lm_head: bool,
    quantize_embeddings: bool,
    quant_source: str,
) -> str:
    """Serialize a single rank's quantized state."""
    artifact = {
        "schema_version": SSD_AWQ_ARTIFACT_VERSION,
        "quant_scheme": "awq_int4",
        "backend": "awq_marlin",
        "model_id": model_id,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "group_size": group_size,
        "use_zero_point": use_zero_point,
        "expected_runtime_dtype": expected_runtime_dtype,
        "quantize_lm_head": quantize_lm_head,
        "quantize_embeddings": quantize_embeddings,
        "quant_source": quant_source,
        "ssd_module_names": sorted(modules.keys()),
        "modules": dict(modules),
    }
    path = _artifact_path(prefix, tp_rank)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(artifact, path)
    return path


def load_awq_artifact(
    prefix: str,
    *,
    tp_rank: int,
    tp_size: int,
    expected_runtime_dtype: str | None = None,
    expected_model_id: str | None = None,
) -> Dict[str, Any]:
    """Load and validate a single rank's artifact.

    Validation matches plan §7.4: exact match required on tp_size / tp_rank,
    model_id (if provided), backend, and expected runtime dtype (if provided).
    """
    path = _artifact_path(prefix, tp_rank)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AWQ artifact not found: {path}")
    art = torch.load(path, map_location="cpu", weights_only=False)

    if art.get("schema_version") != SSD_AWQ_ARTIFACT_VERSION:
        raise ValueError(
            f"SSD AWQ artifact schema_version={art.get('schema_version')} != "
            f"runtime {SSD_AWQ_ARTIFACT_VERSION} ({path})"
        )
    if art["backend"] != "awq_marlin":
        raise ValueError(f"artifact backend {art['backend']!r} != awq_marlin ({path})")
    if art["tp_size"] != tp_size:
        raise ValueError(
            f"artifact tp_size={art['tp_size']} != runtime tp_size={tp_size} ({path})"
        )
    if art["tp_rank"] != tp_rank:
        raise ValueError(
            f"artifact tp_rank={art['tp_rank']} != runtime tp_rank={tp_rank} ({path})"
        )
    if expected_model_id is not None and art.get("model_id") not in (None, expected_model_id):
        raise ValueError(
            f"artifact model_id={art.get('model_id')!r} != runtime {expected_model_id!r} ({path})"
        )
    if expected_runtime_dtype is not None \
            and art.get("expected_runtime_dtype") != expected_runtime_dtype:
        raise ValueError(
            f"artifact expected_runtime_dtype={art.get('expected_runtime_dtype')!r} "
            f"!= runtime {expected_runtime_dtype!r} ({path})"
        )
    return art
