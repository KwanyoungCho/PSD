"""Phase 3a thin adapter: load an external AutoAWQ checkpoint directly into a
live SSD model without going through an on-disk SSD-native artifact.

Primary use: fast iteration / backend validation against a single existing
AutoAWQ HF repo. For production or cold-start speed, prefer the Phase 3b
offline importer + Phase 4 loader combo.
"""
from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, Tuple

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig

from ssd.quant.build import (
    RawAwqTensors,
    concat_packed_awq,
    shard_awq_column_parallel,
    shard_awq_row_parallel,
    build_awq_state,
)


def _read_awq_safetensors_trios(path: str, num_layers: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    from ssd.quant.importer import _collect_autoawq_tensors
    return _collect_autoawq_tensors(path, num_layers, group_size=128)


def load_external_autoawq_into_model(
    model: nn.Module,
    hf_path: str,
    *,
    tp_rank: int,
    tp_size: int,
    verbose: bool = True,
) -> dict:
    """Load AutoAWQ checkpoint → SSD model in memory, per-rank.

    Does the same transforms as the Phase 3b importer but in-memory and
    only for the calling rank (no artifact file written).
    """
    cfg = AutoConfig.from_pretrained(hf_path)
    num_layers = cfg.num_hidden_layers

    group_size = 128
    qc_path = os.path.join(hf_path, "quantize_config.json")
    if os.path.isfile(qc_path):
        with open(qc_path) as f:
            qc = json.load(f)
        group_size = int(qc.get("q_group_size", qc.get("group_size", group_size)))

    trios = _read_awq_safetensors_trios(hf_path, num_layers)

    from ssd.quant.importer import _autoawq_to_raw, _build_layer_packed
    hf_by_base: Dict[str, RawAwqTensors] = {
        name: _autoawq_to_raw(qw, qz, sc, group_size=group_size)
        for name, (qw, qz, sc) in trios.items()
    }
    full_rank = _build_layer_packed(hf_by_base, num_layers)

    # Shard only the relevant rank — full-rank tensors are on CPU, so this is cheap.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    mods = dict(model.named_modules())
    n_attached = 0
    for name, raw in full_rank.items():
        if name.endswith("qkv_proj") or name.endswith("gate_up_proj"):
            shard = shard_awq_column_parallel(raw, tp_rank=tp_rank, tp_size=tp_size)
        else:
            shard = shard_awq_row_parallel(raw, tp_rank=tp_rank, tp_size=tp_size)
        if name not in mods:
            raise KeyError(f"AutoAWQ adapter: SSD model has no module {name!r}")
        state = build_awq_state(shard, device=device)
        mods[name].attach_quant_state(state)
        n_attached += 1

    if verbose:
        print(f"[awq-adapter] rank={tp_rank}/{tp_size}: attached {n_attached} modules "
              f"from {hf_path}", flush=True)
    return {"n_attached": n_attached, "hf_path": hf_path, "group_size": group_size}
