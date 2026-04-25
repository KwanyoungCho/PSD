"""Phase 4 — load an SSD-native AWQ artifact into a live model.

`apply_ssd_awq_artifact(model, prefix, tp_rank, tp_size)`:
  1. Reads `<prefix>.rank{tp_rank}.awq.pt`
  2. For every module entry, builds `AwqQuantState` on the current CUDA device
  3. Calls `module.attach_quant_state(...)` — which drops the meta placeholder
     and switches the module's forward to Marlin.
"""
from __future__ import annotations

import torch
from torch import nn

from ssd.quant.build import RawAwqTensors, build_awq_state
from ssd.quant.io import load_awq_artifact


def apply_ssd_awq_artifact(
    model: nn.Module,
    *,
    prefix: str,
    tp_rank: int,
    tp_size: int,
    expected_runtime_dtype: str | None = None,
    expected_model_id: str | None = None,
    expected_group_size: int | None = None,
    expected_role: str | None = None,
    verbose: bool = True,
) -> dict:
    """Mutate `model` in place by attaching AWQ quant states from the artifact.

    Validation done here (in addition to io.load_awq_artifact's own checks):
      - `expected_role`, if given, must match artifact `model_role` (v1
        artifacts are treated as "target").
      - `expected_group_size`, if given, must match the artifact's
        `group_size` (CLI `--quant_group_size` flows through here).
      - After attaching, every TP linear left on `meta` is reported as a
        hard failure — artifact must be complete.
    """
    artifact = load_awq_artifact(
        prefix,
        tp_rank=tp_rank, tp_size=tp_size,
        expected_runtime_dtype=expected_runtime_dtype,
        expected_model_id=expected_model_id,
        expected_role=expected_role,
    )
    if expected_group_size is not None and artifact["group_size"] != expected_group_size:
        raise ValueError(
            f"SSD AWQ loader: artifact group_size={artifact['group_size']} != "
            f"runtime expectation {expected_group_size}. This usually indicates "
            f"the --quant_group_size CLI flag contradicts the artifact metadata."
        )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    target_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[artifact["expected_runtime_dtype"]]

    mods = dict(model.named_modules())
    n_attached = 0
    missing = []
    for name, payload in artifact["modules"].items():
        if name not in mods:
            missing.append(name)
            continue
        sc = payload["scales"]
        if sc.dtype != target_dtype:
            sc = sc.to(target_dtype)
        bias = payload.get("bias")
        if bias is not None and bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
        raw = RawAwqTensors(
            qweight=payload["qweight"],
            qzeros=payload["qzeros"],
            scales=sc,
            in_features=int(payload["in_features"]),
            out_features=int(payload["out_features"]),
            group_size=int(payload["group_size"]),
            bias=bias,
        )
        state = build_awq_state(raw, device=device)
        mods[name].attach_quant_state(state)
        n_attached += 1

    if missing:
        raise KeyError(
            f"SSD AWQ loader: artifact references {len(missing)} modules "
            f"not present in the model — first few: {missing[:3]}. "
            f"Usually means the artifact was generated from a different "
            f"model architecture or at a different TP configuration."
        )

    # Completeness check (plan §6.3.1, §8.1): after attaching, every TP
    # linear that was constructed with a meta-device placeholder must now
    # have a quant_state. Without this check, a partial artifact (missing
    # some decoder layers or a renamed module) would silently leave a
    # meta weight in place and only crash on the first forward pass.
    from ssd.layers.linear import LinearBase
    unattached = []
    for name, mod in mods.items():
        if not isinstance(mod, LinearBase):
            continue
        w = getattr(mod, "weight", None)
        if w is not None and w.device.type == "meta" and mod.quant_state is None:
            unattached.append(name)
    if unattached:
        raise RuntimeError(
            f"SSD AWQ loader: artifact at '{prefix}' did not provide quant "
            f"state for {len(unattached)} TP linear module(s) that were "
            f"constructed in quant mode. These modules would crash on first "
            f"forward because their weight is still a meta placeholder.\n"
            f"First few missing: {unattached[:5]}\n"
            f"Causes: artifact built from a different model architecture, "
            f"wrong tp_size, or an importer bug that dropped a module."
        )

    if verbose:
        import torch.distributed as dist
        r = dist.get_rank() if dist.is_initialized() else tp_rank
        total_bytes = sum(
            m.quant_state.storage_bytes()
            for m in mods.values()
            if getattr(m, "quant_state", None) is not None
        )
        role = artifact.get("model_role", "target")
        print(
            f"[awq-loader] role={role} rank={r}/{tp_size}: attached {n_attached} modules, "
            f"quant storage ≈ {total_bytes / 1e9:.2f} GB  "
            f"(source={artifact.get('quant_source')}, schema=v{artifact['schema_version']})",
            flush=True,
        )
    return {"n_attached": n_attached, "artifact": artifact}
