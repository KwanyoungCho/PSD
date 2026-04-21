"""Diagnostic helpers for INT8 weight-only quantization.

Gated by env var SSD_INT8_DEBUG=1. Produces:

- H1: original weight sanity (NaN/Inf/absmax) per target module
- H2: post-quantize AQT state (scale, int_data) stats
- H3: layer 1 forward hook — track input/output stats per sub-op to pinpoint
  which op first introduces inf
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn


def _fmt_tensor_stats(t: torch.Tensor) -> str:
    if t is None:
        return "None"
    if not torch.is_tensor(t):
        return f"non-tensor {type(t).__name__}"
    try:
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        if n_nan == 0 and n_inf == 0:
            absmax = t.float().abs().max().item()
            return f"nan=0 inf=0 absmax={absmax:.4g} shape={tuple(t.shape)} dtype={t.dtype}"
        return f"nan={n_nan} inf={n_inf} shape={tuple(t.shape)} dtype={t.dtype}"
    except Exception as e:
        return f"stat_error: {e}"


def check_original_weight(name: str, weight: torch.Tensor, rank: int):
    n_nan = torch.isnan(weight).sum().item()
    n_inf = torch.isinf(weight).sum().item()
    absmax = weight.float().abs().max().item() if (n_nan == 0 and n_inf == 0) else float('nan')
    n_zero_rows = 0
    if weight.dim() == 2:
        row_absmax = weight.float().abs().max(dim=1).values
        n_zero_rows = (row_absmax == 0).sum().item()
    print(
        f"[H1 r{rank}] {name:<48s} orig bf16: nan={n_nan} inf={n_inf} "
        f"absmax={absmax:.4g} zero_rows={n_zero_rows} shape={tuple(weight.shape)}",
        flush=True,
    )


def check_aqt_state(name: str, new_weight: nn.Parameter, rank: int):
    """Inspect the AffineQuantizedTensor internal state."""
    t = new_weight.data
    tname = type(t).__name__
    try:
        ti = getattr(t, 'tensor_impl', None)
        if ti is None:
            print(f"[H2 r{rank}] {name:<48s} no tensor_impl, type={tname}", flush=True)
            return
        int_data = getattr(ti, 'int_data', None)
        scale = getattr(ti, 'scale', None)
        s_stats = _fmt_tensor_stats(scale)
        d_stats = _fmt_tensor_stats(int_data)
        # extra: any scale that is zero or denormal
        zero_s = (scale == 0).sum().item() if scale is not None else -1
        scale_min_abs = scale.float().abs().min().item() if scale is not None and torch.isfinite(scale).all() else float('nan')
        print(
            f"[H2 r{rank}] {name:<48s} AQT type={tname} "
            f"scale({s_stats} zero={zero_s} min_abs={scale_min_abs:.4g})  "
            f"int_data({d_stats})",
            flush=True,
        )
    except Exception as e:
        print(f"[H2 r{rank}] {name:<48s} AQT inspect error: {e}", flush=True)


class _SubOpHook:
    """Forward hook collecting input/output stats."""
    def __init__(self, tag: str, rank: int, call_limit: int = 4):
        self.tag = tag
        self.rank = rank
        self.count = 0
        self.limit = call_limit

    def __call__(self, module, inputs, output):
        if self.count >= self.limit:
            return
        self.count += 1
        try:
            in0 = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) else None
            if isinstance(output, (tuple, list)):
                out0 = output[0]
            else:
                out0 = output
            in_stats = _fmt_tensor_stats(in0)
            out_stats = _fmt_tensor_stats(out0)
            print(
                f"[H3 r{self.rank} call#{self.count}] {self.tag:<40s} "
                f"in=[{in_stats}]  out=[{out_stats}]",
                flush=True,
            )
        except Exception as e:
            print(f"[H3 r{self.rank}] {self.tag} hook error: {e}", flush=True)


def install_layer1_hooks(model: nn.Module, rank: int, target_layer_idx: int = 1, call_limit: int = 4):
    """Register forward hooks on layer {target_layer_idx} submodules.

    Goal: find which sub-op first introduces inf during prefill.
    Hooks fire for the first `call_limit` forward calls per module then stop.
    """
    # Llama layout: model.model.layers[idx].{self_attn, mlp, input_layernorm, post_attention_layernorm}
    try:
        layer = model.model.layers[target_layer_idx]
    except (AttributeError, IndexError) as e:
        print(f"[H3] could not find layer {target_layer_idx}: {e}", flush=True)
        return

    targets = [
        (f"L{target_layer_idx}.input_layernorm", layer.input_layernorm),
        (f"L{target_layer_idx}.self_attn.qkv_proj", layer.self_attn.qkv_proj),
        (f"L{target_layer_idx}.self_attn", layer.self_attn),
        (f"L{target_layer_idx}.self_attn.o_proj", layer.self_attn.o_proj),
        (f"L{target_layer_idx}.post_attention_layernorm", layer.post_attention_layernorm),
        (f"L{target_layer_idx}.mlp.gate_up_proj", layer.mlp.gate_up_proj),
        (f"L{target_layer_idx}.mlp", layer.mlp),
        (f"L{target_layer_idx}.mlp.down_proj", layer.mlp.down_proj),
    ]
    for tag, mod in targets:
        if mod is None:
            continue
        mod.register_forward_hook(_SubOpHook(tag, rank, call_limit))
        print(f"[H3 r{rank}] hook installed: {tag}", flush=True)


def debug_enabled() -> bool:
    return os.environ.get("SSD_INT8_DEBUG", "0") == "1"
