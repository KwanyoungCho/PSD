"""CLI entry point for the Phase 3b offline AWQ importer.

Usage (from ssd/ repo root):

    # RTN W4A16 — no external tool needed, suitable as a baseline and for
    # pipeline validation (plan §16.2 mitigation: "measure MESA accept rate
    # with AWQ vs round-to-nearest").
    python scripts/awq_import.py \
        --model /data2/.../Llama-3.2-1B-Instruct \
        --out   /tmp/awq_artifacts/llama1b \
        --tp    1 \
        --mode  rtn

    # External AutoAWQ checkpoint → SSD-native artifact
    python scripts/awq_import.py \
        --model /path/to/Llama-3.1-8B-AWQ \
        --out   /tmp/awq_artifacts/llama8b \
        --tp    2 \
        --mode  autoawq
"""
import argparse
import sys
import os

# Make sure the ssd package is importable when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SSD_HF_CACHE", "/dev/null")
os.environ.setdefault("SSD_DATASET_DIR", "/dev/null")
os.environ.setdefault("SSD_CUDA_ARCH", os.environ.get("SSD_CUDA_ARCH", "8.6"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to HF model dir")
    p.add_argument("--out", required=True, help="Artifact prefix (no extension)")
    p.add_argument("--tp", type=int, default=1, help="Target TP size")
    p.add_argument("--mode", choices=["rtn", "autoawq"], default="rtn",
                   help="rtn = RTN W4A16 from dense weights; autoawq = repack external AWQ")
    p.add_argument("--group_size", type=int, default=128,
                   help="AWQ group size. For --mode autoawq this must equal the "
                        "checkpoint's q_group_size (auto-detected from quantize_config.json; "
                        "override only if the two disagree).")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                   help="expected runtime dtype (stamped on the artifact)")
    p.add_argument("--base-model", dest="base_model", default=None,
                   help="Absolute path to stamp as the artifact's `model_id`. "
                        "Runtime requires `config.model` to match this path. "
                        "Defaults to --model; override only if the runtime "
                        "config.model points somewhere different (rare).")
    args = p.parse_args()

    from ssd.quant.importer import import_dense_to_ssd_artifact, import_autoawq_to_ssd_artifact

    if args.mode == "rtn":
        written = import_dense_to_ssd_artifact(
            model_path=args.model,
            out_prefix=args.out,
            tp_size=args.tp,
            group_size=args.group_size,
            expected_runtime_dtype=args.dtype,
            base_model_path=args.base_model,
        )
    else:
        written = import_autoawq_to_ssd_artifact(
            model_path=args.model,
            out_prefix=args.out,
            tp_size=args.tp,
            group_size=args.group_size,
            expected_runtime_dtype=args.dtype,
            base_model_path=args.base_model,
        )
    print(f"[awq_import] wrote {len(written)} rank files → {args.out}.rank*.awq.pt")


if __name__ == "__main__":
    main()
