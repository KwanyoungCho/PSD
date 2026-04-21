"""Weight-only quantization hook for SSD target model.

Backends (chosen via `backend` arg):
  - "int4_wo_tile" (DEFAULT): torchao `Int4WeightOnlyConfig(group_size=128)`
    with TensorCoreTiledLayout → tinygemm fast path on Ampere+ (SM 80/86/89).
    ~4× weight memory reduction, faster than dense on small-batch
    verify/decode shapes (where SSD spec spends most time).
  - "int8_wo" : torchao `Int8WeightOnlyConfig`. ~2× memory reduction, but no
    fused int8-weight kernel on SM 86 — dequant + bf16 matmul path, so usually
    slower than dense. Kept for accuracy comparison or other hardware paths.

**Activation dtype compatibility** (important):
  Both backends above are documented (torchao inference docs) as **bf16-activation
  workflows**. On fp16 activation they fail:
    - Int4: API-level assert ("Expected zeros fp16, got bf16")
    - Int8: no assert but produces inf values in layer output (numerically unreliable)
  fp16-checkpoint users must either
    (a) explicitly opt-in to bf16-runtime override (`target_quant_force_bf16_runtime=True`,
        which means "fp16 checkpoint runs on a bf16 runtime" — not a fp16 runtime)
    (b) switch to a fp16-native WO backend (e.g. GemliteUIntXWeightOnlyConfig,
        Marlin) — NOT integrated here yet.
  Without (a) or (b), model_runner raises ValueError at init, surfacing the
  unsupported combination.

Hook contract (confirmed in Phase 0 spike):
  1. Create a dummy `nn.Linear`, copy the float local shard into it.
  2. `quantize_(dummy, <Config>())` — replaces `dummy.weight` with an
     `nn.Parameter` whose `.data` is an `AffineQuantizedTensor`.
  3. `module.weight = dummy.weight` — our custom TP module now holds an
     AQT-backed Parameter. Existing `F.linear(x, self.weight, bias)` calls
     dispatch to quantized kernels via `__torch_dispatch__` (no forward
     code change on the SSD side).

Gated behind `config.target_quant_enabled`. Draft weights are NOT quantized
(only the target model). Weight tying (llama3.py:333-334) is handled by
untying `lm_head.weight` before quantization when `lm_head` quantization is on.
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


def _make_dummy(weight: torch.Tensor, target_dtype: torch.dtype | None) -> nn.Linear:
    assert weight.dim() == 2, f"expect 2D weight, got {weight.shape}"
    out_f, in_f = weight.shape
    dtype = target_dtype if target_dtype is not None else weight.dtype
    dummy = nn.Linear(in_f, out_f, bias=False).to(device=weight.device, dtype=dtype)
    with torch.no_grad():
        dummy.weight.copy_(weight.to(dtype) if weight.dtype != dtype else weight)
    return dummy


def _quantize_weight_to_int8_wo(weight: torch.Tensor, target_dtype: torch.dtype | None = None) -> nn.Parameter:
    """Int8 weight-only via torchao. Per-row scale, dequant + bf16 matmul path.

    Note: On SM 86 (RTX 3090) this path is slower than dense bf16 because there's
    no fused int8-weight + bf16-activation kernel in torchao. Use `int4_wo_tile_packed`
    for actual speedup on small-batch verify/decode.
    """
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    dummy = _make_dummy(weight, target_dtype)
    quantize_(dummy, Int8WeightOnlyConfig())
    return dummy.weight


def _quantize_weight_to_int4_wo(weight: torch.Tensor, target_dtype: torch.dtype | None = None,
                                 group_size: int = 128) -> nn.Parameter:
    """Int4 weight-only via torchao with TensorCoreTiledLayout (default).

    Uses a tinygemm-based fast path on Ampere+ (SM 80/86/89). Measured on
    SM 86 with 8B verify-shape matmuls: 0.25x-1.25x of dense (up to 4x faster
    in AR decode). Prefill shapes are ~3x slower (compute-bound + dequant
    overhead), but prefill is one-shot per sequence so the verify speedup
    dominates the cumulative MESA/spec workload.

    `input_dim` must be divisible by `group_size` (default 128). All Llama
    TP shards we target satisfy this at TP ∈ {2, 4}.
    """
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    out_f, in_f = weight.shape
    assert in_f % group_size == 0, \
        f"INT4 requires in_features ({in_f}) divisible by group_size ({group_size})"
    dummy = _make_dummy(weight, target_dtype)
    quantize_(dummy, Int4WeightOnlyConfig(group_size=group_size))
    return dummy.weight


def apply_quantization_to_target(
    model: nn.Module,
    *,
    quantize_lm_head: bool = False,
    tie_word_embeddings: bool = False,
    verbose: bool = True,
    skip_module_name_substrings: tuple = (),
    backend: str = "int4_wo_tile",
) -> dict:
    """Apply weight-only quantization to target TP linear modules.

    Args:
        model: target model (e.g., LlamaForCausalLM)
        quantize_lm_head: whether to quantize ParallelLMHead. Default False
            because lm_head is a hot path (per-step gather+cat + MESA exit
            logits) and quantizing it costs throughput/accept.
        tie_word_embeddings: `hf_config.tie_word_embeddings`. When True and
            `quantize_lm_head=True`, untie lm_head before quantization to
            avoid corrupting `embed_tokens` which shares the same storage.
        verbose: per-module progress logs.
        skip_module_name_substrings: names containing any substring here are
            left dense (useful for troubleshooting specific modules).
        backend: "int4_wo_tile" (default) or "int8_wo". See module docstring.

    Returns:
        stats dict with counts, byte estimates, and backend label.
    """
    n_tp = 0
    n_lm_head = 0
    bytes_before = 0
    bytes_after_nominal = 0   # int_data-only (nominal = bits × numel / 8)
    bytes_after_actual = 0    # measured: int_data + scale + zero_point storage

    # Optional per-module sanity diagnostics (gated by SSD_INT8_DEBUG=1)
    from ssd.utils.int8_debug import debug_enabled, check_original_weight, check_aqt_state
    _dbg = debug_enabled()
    try:
        import torch.distributed as _dist
        _rank = _dist.get_rank() if _dist.is_initialized() else 0
    except Exception:
        _rank = 0

    # Note: fp16 upcast handling lives in model_runner.py (load-time full upcast)
    # because per-module mixed-dtype introduces activation/weight dtype mismatches
    # in the forward pipeline.
    _q_dtype = None  # None → keep source dtype

    if backend == "int4_wo_tile":
        _q_fn = _quantize_weight_to_int4_wo
        _bytes_per_elem = 0.5  # int4 packed nominal = 0.5 B/elem, ignores scale/zero/layout overhead
        _backend_label = "int4"
        if verbose:
            print(f"[quant] backend=int4_wo_tile (torchao Int4WeightOnlyConfig, tinygemm fast path)", flush=True)
    elif backend == "int8_wo":
        _q_fn = _quantize_weight_to_int8_wo
        _bytes_per_elem = 1.0  # int8 nominal = 1 B/elem, ignores scale overhead
        _backend_label = "int8"
        if verbose:
            print(f"[quant] backend=int8_wo (torchao Int8WeightOnlyConfig)", flush=True)
    else:
        raise ValueError(f"Unknown quant backend: {backend}")

    def _measure_aqt_bytes(p: nn.Parameter) -> int:
        """Best-effort accounting of actual storage bytes for an AQT weight.

        AQT is itself a Tensor subclass with a *logical* shape matching the
        original bf16 weight, so `p.numel() * p.element_size()` gives the
        dense bf16 equivalent, NOT the packed storage. We enumerate tensor-
        valued attributes on `tensor_impl` and sum their *unique* untyped
        storage sizes (deduping aliases like `data` / `packed_weight` /
        `real` that share the same buffer).

        Works across torchao layouts:
          - PlainAQT (Int8WeightOnly): `int_data`, `scale`, `zero_point`
          - TensorCoreTiled (Int4WeightOnly): `packed_weight`, `scale_and_zero`
        """
        data = p.data if isinstance(p, nn.Parameter) else p
        ti = getattr(data, 'tensor_impl', None)
        if ti is None:
            return data.untyped_storage().size() if torch.is_tensor(data) else 0

        seen_ptrs = set()
        total = 0
        for attr in dir(ti):
            if attr.startswith('_'):
                continue
            try:
                t = getattr(ti, attr)
            except Exception:
                continue
            if not torch.is_tensor(t):
                continue
            try:
                storage = t.untyped_storage()
                key = (storage.data_ptr(), storage.size())
            except Exception:
                continue   # some attrs expose meta/fake tensors w/o real storage
            if key in seen_ptrs:
                continue
            seen_ptrs.add(key)
            total += storage.size()
        return total

    # --- Handle lm_head with tie defense first ---
    lm_head = getattr(model, "lm_head", None)
    if quantize_lm_head and isinstance(lm_head, ParallelLMHead):
        # Untie if needed: assigning a new Parameter breaks the data alias.
        if tie_word_embeddings:
            if verbose:
                print(
                    "[quant] tie_word_embeddings=True → untying lm_head before quantize",
                    flush=True,
                )
            lm_head.weight = nn.Parameter(lm_head.weight.data.clone())
        w = lm_head.weight.data
        if _dbg:
            check_original_weight("lm_head", w, _rank)
        bytes_before += w.numel() * w.element_size()
        new_w = _q_fn(w, target_dtype=_q_dtype)
        lm_head.weight = new_w
        if _dbg:
            check_aqt_state("lm_head", new_w, _rank)
        bytes_after_nominal += int(w.numel() * _bytes_per_elem)
        bytes_after_actual += _measure_aqt_bytes(new_w)
        n_lm_head += 1
        if verbose:
            print(
                f"[quant] lm_head {tuple(w.shape)} {w.dtype} → AQT-{_backend_label}",
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
                print(f"[quant] SKIP {name}", flush=True)
            continue
        w = mod.weight.data
        if _dbg:
            check_original_weight(name, w, _rank)
        bytes_before += w.numel() * w.element_size()
        new_w = _q_fn(w, target_dtype=_q_dtype)
        mod.weight = new_w
        if _dbg:
            check_aqt_state(name, new_w, _rank)
        bytes_after_nominal += int(w.numel() * _bytes_per_elem)
        bytes_after_actual += _measure_aqt_bytes(new_w)
        n_tp += 1
        if verbose:
            print(
                f"[quant] {name:<48s} {tuple(w.shape)} {w.dtype} → AQT-{_backend_label}",
                flush=True,
            )

    stats = {
        "n_tp_linear": n_tp,
        "n_lm_head": n_lm_head,
        "bytes_before": bytes_before,
        "bytes_after_nominal": bytes_after_nominal,   # bits × numel / 8 (no overhead)
        "bytes_after_actual": bytes_after_actual,     # measured int_data + scale + zp
        "ratio_nominal": bytes_after_nominal / max(1, bytes_before),
        "ratio_actual": bytes_after_actual / max(1, bytes_before),
    }
    if verbose:
        print(
            f"[quant] summary: backend={_backend_label}, tp_linear={n_tp}, lm_head={n_lm_head}, "
            f"src={bytes_before/1e9:.2f} GB → "
            f"nominal≈{bytes_after_nominal/1e9:.2f} GB (ratio {stats['ratio_nominal']:.2f}), "
            f"actual≈{bytes_after_actual/1e9:.2f} GB (ratio {stats['ratio_actual']:.2f}, incl scale/zp)",
            flush=True,
        )
    return stats


# -------------------------------------------------------------------------
# Phase 5: persistent artifact save/load
# -------------------------------------------------------------------------
def _quantized_module_names(model, quantize_lm_head: bool) -> list:
    names = []
    lm_head = getattr(model, "lm_head", None)
    for name, mod in model.named_modules():
        if mod is lm_head:
            if quantize_lm_head and isinstance(mod, ParallelLMHead):
                names.append(name)
            continue
        if isinstance(mod, ParallelLMHead):
            continue
        if not isinstance(mod, _QUANT_TARGET_TYPES):
            continue
        names.append(name)
    return names


_ARTIFACT_SCHEMA_VERSION = 2   # v2: adds effective_runtime_dtype, original_checkpoint_dtype


def save_quantized_target_artifact(
    model: nn.Module,
    out_path: str,
    *,
    backend: str,
    tp_rank: int,
    tp_size: int,
    quantize_lm_head: bool,
    effective_runtime_dtype: torch.dtype,
    original_checkpoint_dtype: torch.dtype,
    model_id: str | None = None,
    hf_config_snippet: dict | None = None,
    verbose: bool = True,
) -> None:
    """Save per-rank quantized AQT state for target model.

    Artifact is pickle of {
        'schema_version': int,
        'backend': str,
        'tp_rank': int, 'tp_size': int,
        'quantize_lm_head': bool,
        'model_id': str|None,            # path / HF id of original model
        'torch_version': str,
        'torchao_version': str,
        'hf_config': optional snippet,
        'weights': {module_name: AQT Tensor (CPU)},
    }

    NOTE: raw torchao AQT objects are backend-internal and version-fragile.
    Artifact reuse across different torch/torchao versions is NOT supported
    and will fail loud validation in load. For long-term reuse prefer
    regenerating from the original float checkpoint.
    """
    names = _quantized_module_names(model, quantize_lm_head)
    mods_dict = dict(model.named_modules())   # compute once
    weights = {}
    for name in names:
        mod = mods_dict[name]
        w = mod.weight.data
        weights[name] = w.cpu() if w.is_cuda else w

    import importlib
    def _ver(pkg):
        try:
            return importlib.import_module(pkg).__version__
        except Exception:
            return "unknown"

    artifact = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "backend": backend,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "quantize_lm_head": quantize_lm_head,
        # Runtime dtype context (added in v2): distinguishes "fp16 checkpoint
        # loaded into bf16 runtime via force flag" from a true bf16 checkpoint.
        "effective_runtime_dtype": str(effective_runtime_dtype),
        "original_checkpoint_dtype": str(original_checkpoint_dtype),
        "model_id": model_id,
        "torch_version": _ver("torch"),
        "torchao_version": _ver("torchao"),
        "hf_config": hf_config_snippet,
        "weights": weights,
    }
    path = f"{out_path}.rank{tp_rank}.pt"
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(artifact, path)
    if verbose:
        print(f"[quant] saved rank{tp_rank} artifact: {path} ({len(weights)} modules, "
              f"runtime={artifact['effective_runtime_dtype']}, "
              f"orig_ckpt={artifact['original_checkpoint_dtype']}, "
              f"torch={artifact['torch_version']}, torchao={artifact['torchao_version']})", flush=True)


def load_quantized_target_artifact(
    model: nn.Module,
    in_path: str,
    *,
    tp_rank: int,
    tp_size: int,
    expected_backend: str,
    expected_quantize_lm_head: bool | None = None,
    expected_model_id: str | None = None,
    expected_runtime_dtype: torch.dtype | None = None,
    strict_version_match: bool = True,
    verbose: bool = True,
) -> dict:
    """Load per-rank artifact and reassign weights onto target modules.

    Validation:
      - schema_version  : exact match required
      - tp_rank/tp_size : exact match (prevents reuse at different TP)
      - backend         : exact match
      - quantize_lm_head: if expected_quantize_lm_head given, must match
                          (prevents silent extra/missing lm_head quant)
      - model_id        : if expected_model_id given, must match (path)
      - torch/torchao   : if strict_version_match, must match. raw AQT is
                          version-fragile so mismatch can silently corrupt
                          forward. set False only if you accept the risk.
    """
    path = f"{in_path}.rank{tp_rank}.pt"
    artifact = torch.load(path, weights_only=False, map_location="cpu")

    # --- validate ---
    if artifact.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"artifact schema_version {artifact.get('schema_version')} != "
            f"current {_ARTIFACT_SCHEMA_VERSION}"
        )
    if artifact["tp_size"] != tp_size:
        raise ValueError(f"artifact tp_size {artifact['tp_size']} != runtime tp_size {tp_size}")
    if artifact["tp_rank"] != tp_rank:
        raise ValueError(f"artifact tp_rank {artifact['tp_rank']} != runtime tp_rank {tp_rank}")
    if artifact["backend"] != expected_backend:
        raise ValueError(f"artifact backend {artifact['backend']!r} != expected {expected_backend!r}")
    if expected_quantize_lm_head is not None \
            and artifact.get("quantize_lm_head") != expected_quantize_lm_head:
        raise ValueError(
            f"artifact quantize_lm_head={artifact.get('quantize_lm_head')} != "
            f"runtime target_quant_lm_head={expected_quantize_lm_head} "
            "(mismatch would silently change MESA accept/throughput)"
        )
    if expected_model_id is not None \
            and artifact.get("model_id") not in (None, expected_model_id):
        raise ValueError(
            f"artifact model_id {artifact.get('model_id')!r} != runtime {expected_model_id!r}"
        )
    if expected_runtime_dtype is not None:
        art_rt = artifact.get("effective_runtime_dtype")
        if art_rt is not None and art_rt != str(expected_runtime_dtype):
            raise ValueError(
                f"artifact effective_runtime_dtype={art_rt!r} != "
                f"runtime {str(expected_runtime_dtype)!r}. Quantized tensors "
                "(esp. int4 scale_and_zero) are dtype-bound; cross-runtime reuse "
                "silently corrupts matmul output."
            )
    if strict_version_match:
        import importlib
        def _ver(pkg):
            try: return importlib.import_module(pkg).__version__
            except Exception: return "unknown"
        cur_torch, cur_ao = _ver("torch"), _ver("torchao")
        art_torch = artifact.get("torch_version", "unknown")
        art_ao = artifact.get("torchao_version", "unknown")
        if art_torch != cur_torch or art_ao != cur_ao:
            raise ValueError(
                f"version mismatch — artifact (torch={art_torch}, torchao={art_ao}) vs "
                f"runtime (torch={cur_torch}, torchao={cur_ao}). "
                "Raw AQT serialization is backend-internal; regenerate artifact, "
                "or pass strict_version_match=False if you accept silent-corruption risk."
            )

    # --- apply ---
    n_loaded = 0
    mods = dict(model.named_modules())
    for name, w_cpu in artifact["weights"].items():
        if name not in mods:
            raise KeyError(f"artifact has module {name} but current model does not")
        mod = mods[name]
        dev = next(mod.parameters()).device
        w = w_cpu.to(dev) if hasattr(w_cpu, 'to') else w_cpu
        # Always force requires_grad=False for inference safety. If the stored tensor
        # was a Parameter with requires_grad=True, @torch.compile paths can try to
        # build an autograd graph and fail.
        if isinstance(w, nn.Parameter):
            w.requires_grad_(False)
            mod.weight = w
        else:
            mod.weight = nn.Parameter(w, requires_grad=False)
        n_loaded += 1

    if verbose:
        print(f"[quant] loaded rank{tp_rank} artifact: {path} ({n_loaded} modules, "
              f"schema=v{artifact['schema_version']})", flush=True)
    return artifact
