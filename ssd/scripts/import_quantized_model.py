#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssd.quantization.importers import build_importer


def parse_args():
    parser = argparse.ArgumentParser(description="Import model into canonical SSD quantized runtime format.")
    parser.add_argument("--source", required=True, help="Path to source model directory")
    parser.add_argument(
        "--source-format",
        required=True,
        choices=["hf_float"],
        help="Source checkpoint format",
    )
    parser.add_argument("--output", required=True, help="Output directory for canonical quantized artifact")
    parser.add_argument("--tp-size", required=True, type=int, help="Target TP size to export for")
    parser.add_argument(
        "--quant-method",
        default="int8_wo",
        choices=["int8_wo"],
        help="Quantization method for exported artifact",
    )
    parser.add_argument(
        "--scale-dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Scale tensor dtype to store",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    importer = build_importer(args.source_format)
    importer.export(
        source_path=args.source,
        out_dir=args.output,
        tp_size=args.tp_size,
        quant_method=args.quant_method,
        scale_dtype=args.scale_dtype,
    )
    print(
        f"[import_quantized_model] Export complete: source={args.source} "
        f"format={args.source_format} output={args.output} tp_size={args.tp_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
