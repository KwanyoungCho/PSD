from .hf_float import HFFloatImporter


def build_importer(source_format: str):
    if source_format == "hf_float":
        return HFFloatImporter()
    raise ValueError(f"Unsupported source_format: {source_format}")


__all__ = ["HFFloatImporter", "build_importer"]
