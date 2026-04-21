"""INT8 weight-only quantization hook for SSD target model.

Phase 2 eager bring-up. Applies torchao `Int8WeightOnlyConfig` to target
TP linear modules by the contract confirmed in Phase 0:

  1. Create a dummy `nn.Linear`, copy the float local shard into it.
  2. `quantize_(dummy, Int8WeightOnlyConfig())` — swaps `dummy.weight.data`
     for an `AffineQuantizedTensor` while keeping the `nn.Parameter` wrapper.
  3. `module.weight = dummy.weight` — our custom TP module now holds an
     AQT-backed Parameter. Existing `F.linear(x, self.weight, bias)` calls
     dispatch to int8 kernels via `__torch_dispatch__`.

Gated behind `config.target_quant_enabled`. Draft model is never touched.
Weight tying (llama3.py:333-334) is handled by untying `lm_head.weight`
before quantization when `lm_head` quantization is on.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ssd.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from ssd.layers.embed_head import ParallelLMHead


# Target module types for quantization. Explicit allowlist — do NOT match on
# LinearBase, or ReplicatedLinear (unused in Llama) would be pulled in.
_QUANT_TARGET_TYPES = (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    ColumnParallelLinear,  # parent of QKV/Merged; explicit for extra safety
    RowParallelLinear,
)


def _quantize_weight_to_int8_wo(weight: torch.Tensor) -> nn.Parameter:
    """Return an `nn.Parameter` whose .data is an `AffineQuantizedTensor` produced
    by torchao's stable `quantize_()` path via a dummy `nn.Linear` wrapper.
    """
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    assert weight.dim() == 2, f"expect 2D weight, got {weight.shape}"
    out_f, in_f = weight.shape
    dummy = nn.Linear(in_f, out_f, bias=False).to(
        device=weight.device, dtype=weight.dtype
    )
    with torch.no_grad():
        dummy.weight.copy_(weight)
    quantize_(dummy, Int8WeightOnlyConfig())
    return dummy.weight   # nn.Parameter wrapping AffineQuantizedTensor


def apply_int8_weight_only_to_target(
    model: nn.Module,
    *,
    quantize_lm_head: bool = True,
    tie_word_embeddings: bool = False,
    verbose: bool = True,
    skip_module_name_substrings: tuple = (),
) -> dict:
    """Replace local weight tensors of target TP linear modules with INT8
    weight-only AffineQuantizedTensor Parameters.

    Args:
        model: target model (e.g., LlamaForCausalLM)
        quantize_lm_head: whether to quantize ParallelLMHead
        tie_word_embeddings: hf_config.tie_word_embeddings
        verbose: print per-module progress

    Returns:
        stats dict with counts and memory estimate.
    """
    n_tp = 0
    n_lm_head = 0
    bytes_before = 0
    bytes_after = 0

    # --- Handle lm_head with tie defense first ---
    lm_head = getattr(model, "lm_head", None)
    if quantize_lm_head and isinstance(lm_head, ParallelLMHead):
        # Untie if needed: assigning a new Parameter breaks the data alias.
        if tie_word_embeddings:
            if verbose:
                print(
                    "[int8-quant] tie_word_embeddings=True → untying lm_head before quantize",
                    flush=True,
                )
            lm_head.weight = nn.Parameter(lm_head.weight.data.clone())
        w = lm_head.weight.data
        bytes_before += w.numel() * w.element_size()
        new_w = _quantize_weight_to_int8_wo(w)
        lm_head.weight = new_w
        # Count int8 bytes (approx; AQT also holds scale/zp but dominated by int_data)
        bytes_after += w.numel()  # int8 = 1B per element
        n_lm_head += 1
        if verbose:
            print(
                f"[int8-quant] lm_head {tuple(w.shape)} bf16 → AQT-int8",
                flush=True,
            )

    # --- Iterate TP linear modules ---
    n_skipped = 0
    for name, mod in model.named_modules():
        # skip the lm_head (handled above); any nested lm_head wouldn't exist
        if mod is lm_head:
            continue
        # Skip ParallelLMHead subclasses if any other instance slipped in
        if isinstance(mod, ParallelLMHead):
            continue
        if not isinstance(mod, _QUANT_TARGET_TYPES):
            continue
        if any(s in name for s in skip_module_name_substrings):
            n_skipped += 1
            if verbose:
                print(f"[int8-quant] SKIP {name}", flush=True)
            continue
        w = mod.weight.data
        bytes_before += w.numel() * w.element_size()
        new_w = _quantize_weight_to_int8_wo(w)
        mod.weight = new_w
        bytes_after += w.numel()
        n_tp += 1
        if verbose:
            print(
                f"[int8-quant] {name:<48s} {tuple(w.shape)} bf16 → AQT-int8",
                flush=True,
            )

    stats = {
        "n_tp_linear": n_tp,
        "n_lm_head": n_lm_head,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "ratio": bytes_after / max(1, bytes_before),
    }
    if verbose:
        print(
            f"[int8-quant] summary: tp_linear={n_tp}, lm_head={n_lm_head}, "
            f"bf16_bytes={bytes_before/1e9:.2f} GB → int8_bytes≈{bytes_after/1e9:.2f} GB "
            f"(ratio {stats['ratio']:.2f})",
            flush=True,
        )
    return stats
