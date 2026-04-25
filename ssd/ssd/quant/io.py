"""SSD-native AWQ artifact format — plan §7.3 / §7.4.

Schema v2 adds `model_role` so the loader can reject a target artifact
being loaded into a draft model and vice-versa. v1 artifacts are still
accepted (treated as role="target" for backward compatibility with the
first-cut target-only pipeline).

Each per-rank file contains packed AWQ tensors (AutoAWQ layout, not yet
Marlin-repacked — we repack on load because the repack is a cheap CUDA
op and the Marlin layout is version-tied).

File layout (torch.save, pickle):

    {
        "schema_version": SSD_AWQ_ARTIFACT_VERSION,   # 2
        "quant_scheme": "awq_int4",
        "backend": "awq_marlin",
        "model_role": "target" | "draft",             # NEW in v2
        "model_id": str,
        "tp_size": int,
        "tp_rank": int,
        "group_size": 128,
        "use_zero_point": True,
        "expected_runtime_dtype": "float16" | "bfloat16",
        "quantize_lm_head": bool,
        "quantize_embeddings": bool,
        "quant_source": "rtn" | "awq_calibrated" | "external",
        "ssd_module_names": list[str],
        "modules": { module_name: {...}, ... },
    }
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping

import torch

SSD_AWQ_ARTIFACT_VERSION = 2
_SUPPORTED_ARTIFACT_VERSIONS = (1, 2)


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
    model_role: str = "target",
) -> str:
    """Serialize a single rank's quantized state."""
    assert model_role in ("target", "draft"), \
        f"model_role must be target|draft, got {model_role!r}"
    artifact = {
        "schema_version": SSD_AWQ_ARTIFACT_VERSION,
        "quant_scheme": "awq_int4",
        "backend": "awq_marlin",
        "model_role": model_role,
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
    expected_role: str | None = None,
) -> Dict[str, Any]:
    """Load and validate a single rank's artifact.

    Validation (plan §7.4 + role extension):
      - schema_version : must be one of the supported versions
      - backend        : "awq_marlin"
      - tp_size/rank   : exact
      - model_role     : if given, must match. v1 artifacts are treated as
                         "target" for backward compat.
      - model_id       : if given, must match (or artifact may have None)
      - runtime dtype  : if given, must match
    """
    path = _artifact_path(prefix, tp_rank)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AWQ artifact not found: {path}")
    art = torch.load(path, map_location="cpu", weights_only=False)

    if art.get("schema_version") not in _SUPPORTED_ARTIFACT_VERSIONS:
        raise ValueError(
            f"SSD AWQ artifact schema_version={art.get('schema_version')} not in "
            f"supported set {_SUPPORTED_ARTIFACT_VERSIONS} ({path})"
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

    # Role check. v1 artifacts had no role → treat as target for backward compat.
    art_role = art.get("model_role", "target")
    if expected_role is not None and art_role != expected_role:
        raise ValueError(
            f"SSD AWQ loader: artifact model_role={art_role!r} but runtime "
            f"requested role={expected_role!r} ({path}). This prevents loading "
            f"a target artifact into a draft model or vice-versa. Import with "
            f"--role {expected_role!r} or pick the correct artifact."
        )
    # Normalize so callers see a role field even on v1 artifacts.
    art["model_role"] = art_role

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
