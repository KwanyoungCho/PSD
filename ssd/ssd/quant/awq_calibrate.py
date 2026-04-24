"""Offline AWQ calibration — produces an AutoAWQ-format checkpoint from a
dense HF model. Pure-python implementation (no autoawq dependency).

Algorithm (following the AWQ paper, simplified — fixed alpha, no grid search):
  1. For each decoder layer, collect per-input-channel activation magnitudes
     on calibration data via forward hooks.
  2. For foldable groups (q/k/v, gate/up) compute a per-channel scale
        s_i = |x_i|^alpha
     normalized to geometric-mean 1.
  3. Fold into the previous `RMSNorm` by multiplying its weight by 1/s;
     fold into each linear by multiplying its weight columns by s. The
     layer output Y = (Wxs) × (x/s) = Wx is unchanged in fp32.
  4. Quantize every target linear's scaled weight with RTN W4A16 group-128
     and export AutoAWQ-format safetensors + `quantize_config.json`.

Non-foldable linears (`o_proj`, `down_proj`) are RTN-quantized without
an AWQ scale — matches the AWQ paper's practice for these positions
and what autoawq does when `apply_clip=False, apply_scale=True`.

Output format: AutoAWQ-style — a full HF checkpoint where every target
linear's `.weight` has been replaced by `.qweight / .qzeros / .scales`
(AutoAWQ layout), and the preceding LayerNorm's `.weight` has been
divided by the AWQ scale. The existing SSD importer
(`ssd.quant.importer.import_autoawq_to_ssd_artifact`) ingests this
without modification.

Usage (CLI wrapper at `scripts/awq_calibrate.py`):
    python scripts/awq_calibrate.py \\
        --model /data2/.../Llama-3.1-8B-Instruct \\
        --out   /data2/awq_calibrated/llama3_8b_awq \\
        --n-samples 128 --seq-len 512 --alpha 0.5

The output directory is an HF-compatible checkpoint; to produce a
TP-sharded SSD artifact from it:
    python scripts/awq_import.py \\
        --mode autoawq --model /data2/awq_calibrated/llama3_8b_awq \\
        --out /data2/awq_artifacts/llama3_8b_awq --tp 4 --dtype float16
"""
from __future__ import annotations

import json
import os
import shutil
from glob import glob
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm


# --------------- calibration data --------------------------------------------

def get_c4_calibration_samples(tokenizer, n_samples: int, seq_len: int) -> torch.Tensor:
    """Load C4 calibration samples as a [n_samples, seq_len] LongTensor.

    Falls back to pile-style pseudo-random prompts if C4 isn't loadable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train", streaming=True,
        )
        samples = []
        for ex in ds:
            toks = tokenizer(ex["text"], return_tensors="pt", truncation=True,
                              max_length=seq_len).input_ids[0]
            if toks.numel() < seq_len:
                continue
            samples.append(toks[:seq_len])
            if len(samples) >= n_samples:
                break
        if samples:
            return torch.stack(samples, dim=0)
    except Exception as e:
        print(f"[awq-calib] C4 load failed ({type(e).__name__}: {e}); falling back to synthetic")

    # Synthetic fallback — deterministic nonsense tokens. Better than nothing
    # for a smoke test but NOT publishable calibration.
    torch.manual_seed(0)
    V = tokenizer.vocab_size
    return torch.randint(0, V, (n_samples, seq_len), dtype=torch.long)


# --------------- activation collection ---------------------------------------

class _ActObserver:
    """Maintains running mean |x| over dim=-1 (input channels)."""

    def __init__(self):
        self.sum_abs: torch.Tensor | None = None   # [in_features] on cpu-fp32
        self.count: int = 0

    def update(self, x: torch.Tensor) -> None:
        # x: [..., in_features]
        x_abs = x.detach().to(torch.float32).abs().reshape(-1, x.shape[-1])
        s = x_abs.sum(dim=0)
        n = x_abs.shape[0]
        if self.sum_abs is None:
            self.sum_abs = s.cpu()
        else:
            self.sum_abs = self.sum_abs + s.cpu()
        self.count += n

    def mean_abs(self) -> torch.Tensor:
        assert self.sum_abs is not None and self.count > 0
        return self.sum_abs / self.count


def _install_observer(mod: nn.Module, obs: _ActObserver) -> "torch.utils.hooks.RemovableHandle":
    def hook(_mod, inputs, _output):
        obs.update(inputs[0])
    return mod.register_forward_hook(hook)


# --------------- AWQ scale computation ---------------------------------------

def _awq_scale(x_mean_abs: torch.Tensor, alpha: float = 0.5, eps: float = 1e-4) -> torch.Tensor:
    """Compute per-channel AWQ scales s_i = |x_i|^alpha (geometric-mean normalized).

    Returns an fp32 tensor with `prod(s)^(1/n) == 1`.
    """
    s = x_mean_abs.clamp(min=eps).to(torch.float32) ** alpha
    # geometric-mean normalize to keep numerical range tight
    log_s = torch.log(s)
    s = torch.exp(log_s - log_s.mean())
    return s


# --------------- Llama-family foldable groups --------------------------------

def _llama_foldable_groups(model: nn.Module) -> List[Dict]:
    """Return a list of {prev_norm, linears, tag} dicts for each decoder layer.

    Each group's `linears` share input channels and can all be scaled by the
    same AWQ factor folded into `prev_norm`.
    """
    groups = []
    # Walk model.model.layers.N
    layers = model.model.layers
    for i, layer in enumerate(layers):
        # attention: q/k/v share input_layernorm
        groups.append({
            "tag": f"layer{i}.qkv",
            "prev_norm": layer.input_layernorm,
            "linears": [
                ("self_attn.q_proj", layer.self_attn.q_proj),
                ("self_attn.k_proj", layer.self_attn.k_proj),
                ("self_attn.v_proj", layer.self_attn.v_proj),
            ],
            "tap": layer.input_layernorm,    # observer goes on output of this
            "layer_prefix": f"model.layers.{i}",
        })
        # mlp: gate/up share post_attention_layernorm
        groups.append({
            "tag": f"layer{i}.mlp",
            "prev_norm": layer.post_attention_layernorm,
            "linears": [
                ("mlp.gate_proj", layer.mlp.gate_proj),
                ("mlp.up_proj", layer.mlp.up_proj),
            ],
            "tap": layer.post_attention_layernorm,
            "layer_prefix": f"model.layers.{i}",
        })
    return groups


# --------------- scale application (fold) ------------------------------------

def _apply_awq_scales(group: Dict, scales: torch.Tensor) -> None:
    """Fold `scales` into the group: multiply every linear's input-channel
    weights by `scales`, divide the preceding norm's weight by `scales`.

    Y = Wx = (W * diag(s)) * (x / s) = W'x'
    """
    scales = scales.to(group["prev_norm"].weight.device, dtype=group["prev_norm"].weight.dtype)
    # Linears: weight [out_features, in_features], scale along dim=1
    for _, lin in group["linears"]:
        w = lin.weight
        lin.weight.data = (w * scales.view(1, -1)).to(w.dtype)
    # Norm: weight [hidden_size], divide
    n_w = group["prev_norm"].weight
    group["prev_norm"].weight.data = (n_w / scales).to(n_w.dtype)


# --------------- RTN quantization on the (now-scaled) weight ---------------

_TARGET_LINEAR_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def _collect_target_linears(model: nn.Module) -> Dict[str, nn.Linear]:
    out: Dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name.endswith(s) for s in _TARGET_LINEAR_SUFFIXES):
            out[name] = mod
    return out


# --------------- main driver -------------------------------------------------

def run_awq_calibration(
    *,
    model_path: str,
    out_dir: str,
    n_samples: int = 128,
    seq_len: int = 512,
    alpha: float = 0.5,
    group_size: int = 128,
    batch_size: int = 1,
    dtype: torch.dtype | None = None,
    device_map: str = "auto",
) -> None:
    """Full calibration + quantization + AutoAWQ-format export."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[awq-calib] loading {model_path} (device_map={device_map})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    _load_kwargs = {}
    if dtype is not None:
        _load_kwargs["torch_dtype"] = dtype
    if device_map and device_map.lower() not in ("none", "single"):
        _load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **_load_kwargs)
    if "device_map" not in _load_kwargs:
        # Single-GPU fast path (no accelerate dependency). Fits models up to
        # ~12 B on a 24 GB 3090 in fp16/bf16.
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # --- 1. Install observers on every foldable group's tap module.
    groups = _llama_foldable_groups(model)
    observers = [_ActObserver() for _ in groups]
    handles = [_install_observer(g["tap"], obs) for g, obs in zip(groups, observers)]
    print(f"[awq-calib] {len(groups)} foldable groups, hooks installed")

    # --- 2. Run calibration forward passes.
    samples = get_c4_calibration_samples(tokenizer, n_samples, seq_len)
    first_device = next(model.parameters()).device
    n_done = 0
    with torch.inference_mode():
        for i in tqdm(range(0, n_samples, batch_size), desc="[awq-calib] forward"):
            batch = samples[i : i + batch_size].to(first_device)
            model(batch)
            n_done += batch.shape[0]
    print(f"[awq-calib] calibration forward complete: {n_done} samples")

    for h in handles:
        h.remove()

    # --- 3. Compute scales and fold each group.
    print(f"[awq-calib] computing + folding AWQ scales  (alpha={alpha})")
    for g, obs in zip(groups, observers):
        scales = _awq_scale(obs.mean_abs(), alpha=alpha)
        _apply_awq_scales(g, scales)
    # Free observer memory
    del observers

    # --- 4. Quantize every target linear's (now-scaled) weight with RTN.
    print(f"[awq-calib] RTN-quantizing target linears  (group_size={group_size})")
    from ssd.quant.pack import rtn_quantize_w4a16

    quant_payload: Dict[str, Dict[str, torch.Tensor]] = {}
    target_linears = _collect_target_linears(model)
    for name, lin in tqdm(target_linears.items(), desc="[awq-calib] quant"):
        w = lin.weight.data.to("cpu")
        qw, qz, sc, _ = rtn_quantize_w4a16(w, group_size=group_size)
        quant_payload[name] = {"qweight": qw, "qzeros": qz, "scales": sc}

    # --- 5. Build the export state_dict.
    #   - dense keys (embedding, norms, lm_head) are the model's current
    #     weights (which already include the folded AWQ scale on norms).
    #   - target linears are replaced by their qweight/qzeros/scales trio.
    print(f"[awq-calib] building export state_dict")
    new_state: Dict[str, torch.Tensor] = {}
    seen_linear_prefixes = set(quant_payload.keys())

    # iterate in a parameter-path order
    for param_name, param in model.state_dict().items():
        # drop linear `.weight` keys that we replace with qtriple
        if param_name.endswith(".weight"):
            linear_base = param_name[: -len(".weight")]
            if linear_base in seen_linear_prefixes:
                trio = quant_payload[linear_base]
                new_state[f"{linear_base}.qweight"] = trio["qweight"].contiguous()
                new_state[f"{linear_base}.qzeros"] = trio["qzeros"].contiguous()
                new_state[f"{linear_base}.scales"] = trio["scales"].contiguous()
                continue
        new_state[param_name] = param.detach().to("cpu").contiguous()

    # --- 6. Persist as a single-file AutoAWQ-format checkpoint.
    os.makedirs(out_dir, exist_ok=True)
    # Copy static config / tokenizer files
    for name in os.listdir(model_path):
        full = os.path.join(model_path, name)
        if not os.path.isfile(full):
            continue
        if name.endswith((".safetensors", ".bin", ".safetensors.index.json",
                          ".bin.index.json")):
            continue
        shutil.copy2(full, os.path.join(out_dir, name))

    from safetensors.torch import save_file
    save_file(new_state, os.path.join(out_dir, "model.safetensors"))

    with open(os.path.join(out_dir, "quantize_config.json"), "w") as f:
        json.dump(
            {"q_group_size": group_size, "zero_point": True, "w_bit": 4,
             "version": "gemm", "quant_method": "awq",
             "awq_alpha": alpha, "awq_calibration_samples": n_samples,
             "awq_calibration_seq_len": seq_len},
            f, indent=2,
        )
    print(f"[awq-calib] wrote {out_dir}")
    n_linears = len([k for k in quant_payload])
    total_bytes = sum(
        t.numel() * t.element_size()
        for v in quant_payload.values() for t in v.values()
    )
    print(f"[awq-calib] {n_linears} quantized linears, "
          f"packed size ≈ {total_bytes / 1e9:.2f} GB")
