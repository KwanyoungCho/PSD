from dataclasses import dataclass
from glob import glob
import json
import os
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig

import torch

from ssd.quantization.int8_weight_only import quantize_weight_per_channel_int8
from ssd.quantization.runtime_format import QuantManifest, save_manifest, save_rank_state


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported scale dtype: {name}")
    return mapping[name]


@dataclass
class ImportRequest:
    source_path: str
    out_dir: str
    tp_size: int
    quant_method: str
    scale_dtype: str


class HFFloatTensorIndex:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tensor_to_file = self._build_index(model_dir)

    def _build_index(self, model_dir: str) -> dict[str, str]:
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                data = json.load(f)
            return data["weight_map"]

        safetensor_files = glob(os.path.join(model_dir, "*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        tensor_to_file: dict[str, str] = {}
        for file in safetensor_files:
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_to_file[key] = os.path.basename(file)
        return tensor_to_file

    def load_tensor(self, key: str) -> torch.Tensor:
        if key not in self.tensor_to_file:
            raise KeyError(f"Tensor key not found: {key}")
        file_name = self.tensor_to_file[key]
        file_path = os.path.join(self.model_dir, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)


class HFFloatImporter:
    source_format = "hf_float"

    def inspect(self, source_path: str) -> dict:
        cfg = AutoConfig.from_pretrained(source_path)
        return {
            "model_type": cfg.model_type,
            "num_hidden_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "vocab_size": cfg.vocab_size,
        }

    def export(
        self,
        source_path: str,
        out_dir: str,
        tp_size: int,
        quant_method: str,
        scale_dtype: str = "fp16",
    ) -> None:
        if quant_method != "int8_wo":
            raise ValueError(f"Unsupported quant_method for hf_float importer: {quant_method}")

        cfg = AutoConfig.from_pretrained(source_path)
        if cfg.model_type != "llama":
            raise ValueError(f"Only Llama is supported in v1, got {cfg.model_type}")

        request = ImportRequest(
            source_path=source_path,
            out_dir=out_dir,
            tp_size=tp_size,
            quant_method=quant_method,
            scale_dtype=scale_dtype,
        )
        tensor_index = HFFloatTensorIndex(source_path)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifest = QuantManifest(
            format="ssd_int8_wo_v1",
            model_family="llama",
            source_model=source_path,
            source_format=self.source_format,
            tp_size=tp_size,
            quant_method=quant_method,
            scheme="per_channel_symmetric",
            scale_dtype=scale_dtype,
            target_only=True,
            skip_embed=True,
            skip_lm_head=True,
        )
        save_manifest(manifest, out_path)

        for rank in range(tp_size):
            state = self._build_rank_state(request, cfg, tensor_index, rank)
            save_rank_state(out_path / f"rank_{rank}", state)

    def _build_rank_state(
        self,
        request: ImportRequest,
        cfg,
        tensor_index: HFFloatTensorIndex,
        rank: int,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        scale_dtype = _dtype_from_name(request.scale_dtype)

        for layer_idx in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            q = tensor_index.load_tensor(f"{prefix}.self_attn.q_proj.weight")
            k = tensor_index.load_tensor(f"{prefix}.self_attn.k_proj.weight")
            v = tensor_index.load_tensor(f"{prefix}.self_attn.v_proj.weight")
            o = tensor_index.load_tensor(f"{prefix}.self_attn.o_proj.weight")
            gate = tensor_index.load_tensor(f"{prefix}.mlp.gate_proj.weight")
            up = tensor_index.load_tensor(f"{prefix}.mlp.up_proj.weight")
            down = tensor_index.load_tensor(f"{prefix}.mlp.down_proj.weight")

            q_local = self._shard_colwise(q, rank, request.tp_size)
            k_local = self._shard_colwise(k, rank, request.tp_size)
            v_local = self._shard_colwise(v, rank, request.tp_size)
            o_local = self._shard_rowwise(o, rank, request.tp_size)
            gate_local = self._shard_colwise(gate, rank, request.tp_size)
            up_local = self._shard_colwise(up, rank, request.tp_size)
            down_local = self._shard_rowwise(down, rank, request.tp_size)

            qkv_local = torch.cat([q_local, k_local, v_local], dim=0).contiguous()
            gate_up_local = torch.cat([gate_local, up_local], dim=0).contiguous()

            qkv_q, qkv_s = quantize_weight_per_channel_int8(qkv_local, scale_dtype=scale_dtype)
            o_q, o_s = quantize_weight_per_channel_int8(o_local, scale_dtype=scale_dtype)
            gu_q, gu_s = quantize_weight_per_channel_int8(gate_up_local, scale_dtype=scale_dtype)
            down_q, down_s = quantize_weight_per_channel_int8(down_local, scale_dtype=scale_dtype)

            state[f"{prefix}.self_attn.qkv_proj.qweight"] = qkv_q.cpu()
            state[f"{prefix}.self_attn.qkv_proj.scales"] = qkv_s.cpu()
            state[f"{prefix}.self_attn.o_proj.qweight"] = o_q.cpu()
            state[f"{prefix}.self_attn.o_proj.scales"] = o_s.cpu()
            state[f"{prefix}.mlp.gate_up_proj.qweight"] = gu_q.cpu()
            state[f"{prefix}.mlp.gate_up_proj.scales"] = gu_s.cpu()
            state[f"{prefix}.mlp.down_proj.qweight"] = down_q.cpu()
            state[f"{prefix}.mlp.down_proj.scales"] = down_s.cpu()

        return state

    @staticmethod
    def _shard_colwise(weight: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
        if weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Colwise shard requires out_features divisible by tp_size: "
                f"shape={tuple(weight.shape)}, tp_size={tp_size}"
            )
        shard = weight.shape[0] // tp_size
        start = rank * shard
        return weight.narrow(0, start, shard).contiguous()

    @staticmethod
    def _shard_rowwise(weight: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
        if weight.shape[1] % tp_size != 0:
            raise ValueError(
                f"Rowwise shard requires in_features divisible by tp_size: "
                f"shape={tuple(weight.shape)}, tp_size={tp_size}"
            )
        shard = weight.shape[1] // tp_size
        start = rank * shard
        return weight.narrow(1, start, shard).contiguous()
