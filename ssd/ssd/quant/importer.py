"""Phase 3b offline importer — CPU-only.

Given a dense HF checkpoint (safetensors), this module RTN-quantizes every
target linear weight, packs into AutoAWQ layout, concatenates qkv + gate_up
in the correct order, TP-shards per rank, and emits SSD-native
`<prefix>.rank{r}.awq.pt` files.

It can also read an AutoAWQ-format checkpoint (qweight/qzeros/scales present)
and skip the RTN step — used by Phase 3a's thin adapter.

This file is import-time CPU only; no CUDA, no SSD model construction.
Run as a CLI or import `import_dense_to_ssd_artifact` / `import_autoawq_to_ssd_artifact`.
"""
from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open
from transformers import AutoConfig

from ssd.quant.build import (
    RawAwqTensors,
    concat_packed_awq,
    shard_awq_column_parallel,
    shard_awq_row_parallel,
)
from ssd.quant.pack import rtn_quantize_w4a16
from ssd.quant.io import save_awq_artifact
from ssd.quant.naming import llama_layer_ssd_modules, ShardMode, hf_linear_to_ssd


def _iter_safetensor_keys(model_path: str) -> Iterable[Tuple[str, str]]:
    """Yield (file_path, key) for every tensor key in the model dir."""
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors files at {model_path}")
    for f in files:
        with safe_open(f, "pt", "cpu") as sf:
            for k in sf.keys():
                yield f, k


def _read_tensor(model_path: str, key: str) -> torch.Tensor:
    """Read a single named tensor from a model dir (opens only the right file)."""
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    for f in files:
        with safe_open(f, "pt", "cpu") as sf:
            if key in sf.keys():
                return sf.get_tensor(key)
    raise KeyError(f"{key} not found in {model_path}")


def _collect_dense_linear_weights(
    model_path: str,
    num_layers: int,
) -> Dict[str, torch.Tensor]:
    """Collect HF dense linear weights relevant to Llama projections.

    Returns a dict keyed by HF name without the `.weight` suffix.
    """
    wanted = set()
    for i in range(num_layers):
        wanted.update({
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj",
        })

    out: Dict[str, torch.Tensor] = {}
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    for f in files:
        with safe_open(f, "pt", "cpu") as sf:
            keys = sf.keys()
            for k in keys:
                if not k.endswith(".weight"):
                    continue
                base = k[: -len(".weight")]
                if base in wanted:
                    out[base] = sf.get_tensor(k)
    missing = wanted - set(out.keys())
    if missing:
        raise KeyError(
            f"missing {len(missing)} expected linear weights in {model_path}; "
            f"examples: {sorted(missing)[:3]}"
        )
    return out


def _collect_autoawq_tensors(
    model_path: str,
    num_layers: int,
    group_size: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collect AutoAWQ (qweight, qzeros, scales) trios per HF linear.

    Returns dict keyed by HF base name.
    """
    need_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    found: Dict[str, Dict[str, torch.Tensor]] = {}
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    for f in files:
        with safe_open(f, "pt", "cpu") as sf:
            for k in sf.keys():
                if not any(k.endswith(f"{s}.qweight")
                           or k.endswith(f"{s}.qzeros")
                           or k.endswith(f"{s}.scales")
                           for s in need_suffixes):
                    continue
                base, _, field = k.rpartition(".")
                found.setdefault(base, {})[field] = sf.get_tensor(k)
    for base, trio in found.items():
        if not {"qweight", "qzeros", "scales"}.issubset(trio.keys()):
            continue
        out[base] = (trio["qweight"], trio["qzeros"], trio["scales"])
    return out


def _dense_to_raw(
    w: torch.Tensor, group_size: int
) -> RawAwqTensors:
    """RTN-quantize a dense [out, in] weight → AWQ-layout RawAwqTensors."""
    out_f, in_f = w.shape
    qw, qz, sc, _ = rtn_quantize_w4a16(w, group_size=group_size)
    return RawAwqTensors(
        qweight=qw, qzeros=qz, scales=sc,
        in_features=in_f, out_features=out_f, group_size=group_size,
    )


def _autoawq_to_raw(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> RawAwqTensors:
    in_f, out_cols = qweight.shape
    pack = 8
    out_f = out_cols * pack
    return RawAwqTensors(
        qweight=qweight.contiguous(),
        qzeros=qzeros.contiguous(),
        scales=scales.contiguous(),
        in_features=in_f, out_features=out_f, group_size=group_size,
    )


def _build_layer_packed(
    hf_by_base: Dict[str, RawAwqTensors],
    num_layers: int,
) -> Dict[str, RawAwqTensors]:
    """Pack q/k/v and gate/up, keyed by SSD module name (full-rank)."""
    out: Dict[str, RawAwqTensors] = {}
    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # QKV pack — order q → k → v (plan §9.3.1)
        q = hf_by_base[f"{prefix}.self_attn.q_proj"]
        k = hf_by_base[f"{prefix}.self_attn.k_proj"]
        v = hf_by_base[f"{prefix}.self_attn.v_proj"]
        out[f"{prefix}.self_attn.qkv_proj"] = concat_packed_awq([q, k, v])

        # o_proj: pass through
        out[f"{prefix}.self_attn.o_proj"] = hf_by_base[f"{prefix}.self_attn.o_proj"]

        # gate_up pack — order gate → up (plan §9.3.1)
        gate = hf_by_base[f"{prefix}.mlp.gate_proj"]
        up = hf_by_base[f"{prefix}.mlp.up_proj"]
        out[f"{prefix}.mlp.gate_up_proj"] = concat_packed_awq([gate, up])

        # down_proj: pass through
        out[f"{prefix}.mlp.down_proj"] = hf_by_base[f"{prefix}.mlp.down_proj"]
    return out


def _shard_and_serialize(
    full_rank_modules: Dict[str, RawAwqTensors],
    tp_size: int,
) -> list[Dict[str, Dict[str, torch.Tensor | int]]]:
    """Apply column/row sharding per rank and return a per-rank dict-of-dicts."""
    from ssd.quant.naming import _LLAMA_LINEAR_MAP as _  # import side-effect only

    result: list[Dict[str, Dict[str, torch.Tensor | int]]] = [{} for _ in range(tp_size)]
    for name, raw in full_rank_modules.items():
        # Determine shard mode from the module's SSD suffix.
        if name.endswith("qkv_proj") or name.endswith("gate_up_proj"):
            mode = ShardMode.COLUMN
        elif name.endswith("o_proj") or name.endswith("down_proj"):
            mode = ShardMode.ROW
        else:
            raise ValueError(f"unknown SSD module suffix: {name}")
        for r in range(tp_size):
            if mode == ShardMode.COLUMN:
                shard = shard_awq_column_parallel(raw, tp_rank=r, tp_size=tp_size)
            else:
                shard = shard_awq_row_parallel(raw, tp_rank=r, tp_size=tp_size)
            result[r][name] = {
                "qweight": shard.qweight.contiguous(),
                "qzeros": shard.qzeros.contiguous(),
                "scales": shard.scales.contiguous(),
                "in_features": shard.in_features,
                "out_features": shard.out_features,
                "group_size": shard.group_size,
                "bias": shard.bias.contiguous() if shard.bias is not None else None,
            }
    return result


def import_dense_to_ssd_artifact(
    *,
    model_path: str,
    out_prefix: str,
    tp_size: int = 1,
    group_size: int = 128,
    expected_runtime_dtype: str = "float16",
    quantize_lm_head: bool = False,
    quantize_embeddings: bool = False,
    base_model_path: str | None = None,
) -> list[str]:
    """Dense HF checkpoint → SSD-native AWQ artifact (RTN W4A16).

    Args:
        model_path: dense checkpoint directory to quantize.
        base_model_path: what to stamp as `model_id` in the artifact. At
            runtime the loader requires the runtime's `config.model`
            absolute path to match. Defaults to `model_path` — for RTN
            these are the same.
    """
    cfg = AutoConfig.from_pretrained(model_path)
    num_layers = cfg.num_hidden_layers
    print(f"[importer] RTN W4A16 importer  model={model_path}  layers={num_layers}  tp={tp_size}")

    dense = _collect_dense_linear_weights(model_path, num_layers)
    hf_by_base: Dict[str, RawAwqTensors] = {}
    for name, w in dense.items():
        hf_by_base[name] = _dense_to_raw(w, group_size=group_size)
    del dense  # free

    full_rank = _build_layer_packed(hf_by_base, num_layers)
    del hf_by_base

    per_rank = _shard_and_serialize(full_rank, tp_size=tp_size)
    del full_rank

    model_id = os.path.abspath(base_model_path or model_path)
    written = []
    for r in range(tp_size):
        p = save_awq_artifact(
            prefix=out_prefix, tp_rank=r, tp_size=tp_size,
            modules=per_rank[r],
            model_id=model_id,
            group_size=group_size, use_zero_point=True,
            expected_runtime_dtype=expected_runtime_dtype,
            quantize_lm_head=quantize_lm_head,
            quantize_embeddings=quantize_embeddings,
            quant_source="rtn",
        )
        print(f"[importer] wrote rank={r}: {p}  ({len(per_rank[r])} modules)")
        written.append(p)
    return written


def import_autoawq_to_ssd_artifact(
    *,
    model_path: str,
    out_prefix: str,
    tp_size: int = 1,
    group_size: int = 128,
    expected_runtime_dtype: str = "float16",
    quantize_lm_head: bool = False,
    quantize_embeddings: bool = False,
    base_model_path: str | None = None,
) -> list[str]:
    """AutoAWQ HF checkpoint → SSD-native AWQ artifact (same packing, no re-quant).

    Args:
        model_path: directory of an AutoAWQ-format HF checkpoint.
            Contains `{config.json, quantize_config.json, *.safetensors}`.
            Linear weights live as `.qweight / .qzeros / .scales` keys;
            embeddings, lm_head and norms stay dense as `.weight` keys.
        base_model_path: what to stamp as `model_id` in the artifact. At
            runtime the loader requires `config.model` absolute path to
            match. Typically this is **the AutoAWQ dir itself** because
            that's where the dense embeddings/lm_head/norms live and
            the runtime's dense loader will open it. Defaults to
            `model_path`.

    Hard requirements (fail-loudly, plan §5.4):
        - quantize_config.json present (else `RuntimeError`)
        - group_size matches the runtime default (128 unless overridden)
        - zero_point == True   (Marlin uint4 path requires zero-point AWQ)
        - w_bit == 4           (Marlin uint4 path is 4-bit only)
    """
    cfg = AutoConfig.from_pretrained(model_path)
    num_layers = cfg.num_hidden_layers
    print(f"[importer] AutoAWQ importer  model={model_path}  layers={num_layers}  tp={tp_size}")

    # AWQ quant parameters are stored in one of two places depending on the
    # source tool:
    #   - llm-awq / our awq_calibrate.py: `quantize_config.json`
    #       {"q_group_size": ..., "w_bit": ..., "zero_point": ...}
    #   - AutoAWQ (casper-hansen): embeds `config.json["quantization_config"]`
    #       {"group_size": ..., "bits": ..., "zero_point": ..., "quant_method": "awq"}
    # Both are accepted. Key names are normalized here.
    qc = None
    qc_path = os.path.join(model_path, "quantize_config.json")
    if os.path.isfile(qc_path):
        with open(qc_path) as f:
            qc = json.load(f)
    else:
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg_all = json.load(f)
            qc = cfg_all.get("quantization_config")
    if qc is None:
        raise RuntimeError(
            f"AutoAWQ importer: no quantize_config.json and no "
            f"quantization_config in config.json at {model_path}. "
            f"Expected one of these as the source of group_size / zero_point / "
            f"w_bit. If your external toolchain didn't produce one, write a "
            f'{{"q_group_size": 128, "zero_point": true, "w_bit": 4}} by hand.'
        )
    ext_group_size = int(qc.get("q_group_size", qc.get("group_size", -1)))
    ext_zero_point = qc.get("zero_point", qc.get("use_zero_point", None))
    ext_w_bit = int(qc.get("w_bit", qc.get("bits", -1)))
    print(f"[importer]   quantize_config.json: group_size={ext_group_size}  "
          f"zero_point={ext_zero_point}  w_bit={ext_w_bit}")

    if ext_w_bit != 4:
        raise RuntimeError(
            f"AutoAWQ importer: w_bit={ext_w_bit} not supported. The Marlin "
            f"uint4 runtime path is 4-bit only. Re-quantize with w_bit=4 or "
            f"add a new runtime backend."
        )
    if ext_zero_point is not True:
        raise RuntimeError(
            f"AutoAWQ importer: zero_point={ext_zero_point} not supported. "
            f"The Marlin uint4 runtime path requires zero-point AWQ "
            f"(is_zp_float=False, b_q_type=uint4). Symmetric / zp-free AWQ "
            f"checkpoints need a different backend (e.g. GPTQ uint4b8 via "
            f"gptq_marlin_repack)."
        )
    if group_size != ext_group_size:
        if group_size != 128:
            # Explicit override conflicts with the checkpoint.
            raise RuntimeError(
                f"AutoAWQ importer: CLI --group_size={group_size} != "
                f"checkpoint group_size={ext_group_size}. Pass the correct "
                f"--group_size or omit it to use the checkpoint's value."
            )
        print(f"[importer]   using group_size={ext_group_size} from checkpoint")
        group_size = ext_group_size

    raw_trios = _collect_autoawq_tensors(model_path, num_layers, group_size)
    hf_by_base: Dict[str, RawAwqTensors] = {}
    for name, (qw, qz, sc) in raw_trios.items():
        hf_by_base[name] = _autoawq_to_raw(qw, qz, sc, group_size=group_size)

    full_rank = _build_layer_packed(hf_by_base, num_layers)
    per_rank = _shard_and_serialize(full_rank, tp_size=tp_size)

    model_id = os.path.abspath(base_model_path or model_path)
    written = []
    for r in range(tp_size):
        p = save_awq_artifact(
            prefix=out_prefix, tp_rank=r, tp_size=tp_size,
            modules=per_rank[r],
            model_id=model_id,
            group_size=group_size, use_zero_point=True,
            expected_runtime_dtype=expected_runtime_dtype,
            quantize_lm_head=quantize_lm_head,
            quantize_embeddings=quantize_embeddings,
            quant_source="awq_calibrated",
        )
        print(f"[importer] wrote rank={r}: {p}")
        written.append(p)
    return written
