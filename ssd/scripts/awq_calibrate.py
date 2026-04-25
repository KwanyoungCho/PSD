"""[EXPERIMENTAL] CLI for the simplified self-contained AWQ calibrator.

This is a fixed-α (no grid-search), no-clip-search variant of AWQ written
without an external dependency. It exists primarily as a fallback for
models where AutoAWQ trips a corner case (e.g. layerskip-llama2-70B in
our experiments).

**For production use prefer `scripts/awq_calibrate_autoawq.py`** — that
path uses the reference AutoAWQ implementation (grid-search α, clip
search, smoothing) and produces higher-quality scales.

Usage:
    python scripts/awq_calibrate.py \\
        --model /path/to/Llama-3.1-8B-Instruct \\
        --out   /path/to/awq_calibrated/llama3_8b \\
        --n-samples 128 --seq-len 512 --alpha 0.5 \\
        --device-map auto --dtype float16 --max-gpu-memory 18GiB
"""
import argparse
import os
import sys


# Make the ssd package importable from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SSD_HF_CACHE", "/dev/null")
os.environ.setdefault("SSD_DATASET_DIR", "/dev/null")
os.environ.setdefault("SSD_CUDA_ARCH", os.environ.get("SSD_CUDA_ARCH", "8.6"))


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF dense model dir")
    p.add_argument("--out", required=True, help="output dir (AutoAWQ-format checkpoint)")
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="AWQ scale exponent s_i = |x_i|^alpha (paper default 0.5)")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--device-map", default="auto",
                   help="HF device_map. 'auto' auto-shards a large model across visible GPUs.")
    p.add_argument("--max-gpu-memory", default=None,
                   help='Per-GPU cap for accelerate (e.g. "18GiB"). Leaves '
                        'activation headroom.')
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    from ssd.quant.awq_calibrate import run_awq_calibration
    run_awq_calibration(
        model_path=args.model,
        out_dir=args.out,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        alpha=args.alpha,
        group_size=args.group_size,
        batch_size=args.batch_size,
        dtype=dtype,
        device_map=args.device_map,
        max_gpu_memory=args.max_gpu_memory,
    )


if __name__ == "__main__":
    main()
