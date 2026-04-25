"""Phase 3a thin adapter — load an external AutoAWQ checkpoint directly into
a live SSD model without going through an on-disk SSD-native artifact.

Validation policy (after review feedback):
  - mandatory `quantize_config.json` OR `config.json["quantization_config"]`
  - reject `w_bit != 4` / `zero_point != True` (Marlin uint4 only)
  - post-attach completeness check (every meta-mode TP linear must be
    attached, else hard-fail before warmup)
  - per-rank shape validation through the existing `attach_quant_state`

Production-grade calibrate→artifact→load remains the SSD-native artifact
path (`ssd.quant.loader.apply_ssd_awq_artifact`); this direct-load path is
useful for one-off backend validation and ad-hoc experiments.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch
from torch import nn
from transformers import AutoConfig

from ssd.quant.build import (
    RawAwqTensors,
    shard_awq_column_parallel,
    shard_awq_row_parallel,
    build_awq_state,
)


def _read_quant_config(model_path: str) -> dict:
    """Same fallback policy as `importer.py::import_autoawq_to_ssd_artifact`.

    Reads `quantize_config.json` if present, else `config.json["quantization_config"]`.
    Hard-fails when neither has the mandatory fields.
    """
    qc = None
    qc_path = os.path.join(model_path, "quantize_config.json")
    if os.path.isfile(qc_path):
        with open(qc_path) as f:
            qc = json.load(f)
    else:
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                qc = json.load(f).get("quantization_config")
    if qc is None:
        raise RuntimeError(
            f"AutoAWQ adapter: no quantize_config.json and no "
            f"quantization_config in config.json at {model_path}. Provide one "
            f'with at minimum {{"q_group_size": 128, "zero_point": true, "w_bit": 4}}.'
        )

    group_size = int(qc.get("q_group_size", qc.get("group_size", -1)))
    zero_point = qc.get("zero_point", qc.get("use_zero_point", None))
    w_bit = int(qc.get("w_bit", qc.get("bits", -1)))

    if w_bit != 4:
        raise RuntimeError(
            f"AutoAWQ adapter: w_bit={w_bit} not supported (Marlin uint4 only). "
            f"Re-quantize with w_bit=4 or use a different backend."
        )
    if zero_point is not True:
        raise RuntimeError(
            f"AutoAWQ adapter: zero_point={zero_point} not supported "
            f"(Marlin uint4 path requires zero-point AWQ). Symmetric / "
            f"zp-free AWQ checkpoints need a different backend (e.g. GPTQ "
            f"uint4b8 via gptq_marlin_repack)."
        )
    if group_size <= 0:
        raise RuntimeError(
            f"AutoAWQ adapter: missing/invalid group_size in {model_path}"
        )
    return {"group_size": group_size, "zero_point": zero_point, "w_bit": w_bit}


def load_external_autoawq_into_model(
    model: nn.Module,
    hf_path: str,
    *,
    tp_rank: int,
    tp_size: int,
    expected_role: str = "target",
    verbose: bool = True,
) -> dict:
    """Load AutoAWQ checkpoint → SSD model in memory, per-rank.

    Mirrors what the SSD-native artifact loader does, but reads the external
    HF checkpoint directly. Includes the same hard-fail validation surface
    as the offline importer + the completeness check.

    `expected_role` is passed through to the on-attach completeness check;
    today the AutoAWQ format does not embed a role, so the only effect is
    the diagnostic message printed at attach time.
    """
    cfg = AutoConfig.from_pretrained(hf_path)
    num_layers = cfg.num_hidden_layers

    qc = _read_quant_config(hf_path)
    group_size = qc["group_size"]
    if verbose:
        print(f"[awq-adapter] role={expected_role} {hf_path}  "
              f"group_size={group_size} zero_point={qc['zero_point']} "
              f"w_bit={qc['w_bit']}", flush=True)

    from ssd.quant.importer import _collect_autoawq_tensors, _autoawq_to_raw, _build_layer_packed
    trios = _collect_autoawq_tensors(hf_path, num_layers, group_size)
    if not trios:
        raise RuntimeError(
            f"AutoAWQ adapter: found no .qweight/.qzeros/.scales trios in "
            f"{hf_path}. Wrong directory, or non-AutoAWQ format."
        )
    hf_by_base: Dict[str, RawAwqTensors] = {
        name: _autoawq_to_raw(qw, qz, sc, group_size=group_size)
        for name, (qw, qz, sc) in trios.items()
    }
    full_rank = _build_layer_packed(hf_by_base, num_layers)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    mods = dict(model.named_modules())
    n_attached = 0
    for name, raw in full_rank.items():
        if name.endswith("qkv_proj") or name.endswith("gate_up_proj"):
            shard = shard_awq_column_parallel(raw, tp_rank=tp_rank, tp_size=tp_size)
        else:
            shard = shard_awq_row_parallel(raw, tp_rank=tp_rank, tp_size=tp_size)
        if name not in mods:
            raise KeyError(
                f"AutoAWQ adapter: SSD model has no module {name!r}. "
                f"Architecture mismatch between hf_path and the SSD model."
            )
        state = build_awq_state(shard, device=device)
        mods[name].attach_quant_state(state)
        n_attached += 1

    # Completeness check (parity with `apply_ssd_awq_artifact`): no TP linear
    # may remain on the meta device after attach.
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
            f"AutoAWQ adapter: {len(unattached)} TP linear module(s) were "
            f"constructed in quant mode but received no quant state from "
            f"{hf_path}. First few: {unattached[:5]}. The external "
            f"checkpoint is incomplete for this SSD model architecture."
        )

    if verbose:
        print(f"[awq-adapter] role={expected_role} rank={tp_rank}/{tp_size}: "
              f"attached {n_attached} modules from {hf_path}", flush=True)
    return {"n_attached": n_attached, "hf_path": hf_path, "group_size": group_size}
