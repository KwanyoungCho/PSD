from dataclasses import asdict, dataclass
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


@dataclass
class QuantManifest:
    format: str
    model_family: str
    source_model: str
    source_format: str
    tp_size: int
    quant_method: str
    scheme: str
    scale_dtype: str
    target_only: bool
    skip_embed: bool
    skip_lm_head: bool
    version: int = 1


def save_manifest(manifest: QuantManifest, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "manifest.json").open("w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)


def load_manifest(model_dir: str | Path) -> QuantManifest:
    model_path = Path(model_dir)
    with (model_path / "manifest.json").open() as f:
        return QuantManifest(**json.load(f))


def save_rank_state(rank_dir: str | Path, state_dict: dict) -> None:
    rank_path = Path(rank_dir)
    rank_path.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(rank_path / "model.safetensors"))


def load_rank_state(rank_dir: str | Path) -> dict:
    rank_path = Path(rank_dir)
    return load_file(str(rank_path / "model.safetensors"))


def is_quantized_model_dir(model_dir: str | Path) -> bool:
    model_path = Path(model_dir)
    manifest_path = model_path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = load_manifest(model_path)
    except Exception:
        return False
    return manifest.format.startswith("ssd_int8_wo_")
