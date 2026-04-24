"""Offline AWQ calibration via AutoAWQ (casper-hansen/AutoAWQ).

Produces an AutoAWQ-format checkpoint that our existing SSD importer
(`scripts/awq_import.py --mode autoawq`) ingests.

**Environment**: must be run in the separate `awq-quant` conda env — AutoAWQ
pins its own torch / transformers versions that conflict with the `ssd`
engine env. The split mirrors how sglang / vllm baselines are kept in
separate envs.

Usage:
    /home/chokwans99/anaconda3/envs/awq-quant/bin/python \\
      scripts/awq_calibrate_autoawq.py \\
      --model /data2/.../layerskip-codellama-34B \\
      --out   /data2/chokwans99/awq_calibrated/codellama34b \\
      --w-bit 4 --group-size 128 --version GEMM
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF dense model dir")
    p.add_argument("--out", required=True, help="output dir (AutoAWQ-format checkpoint)")
    p.add_argument("--w-bit", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--zero-point", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=True)
    p.add_argument("--version", default="GEMM",
                   help="AutoAWQ pack version — GEMM (default, fp16 activation) or GEMV/GEMVFast")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--max-calib-samples", type=int, default=512,
                   help="AutoAWQ default. Use 128 for a quick run.")
    p.add_argument("--max-calib-seq-len", type=int, default=512)
    p.add_argument("--n-parallel-calib-samples", type=int, default=None,
                   help="batch size during calibration forward. None = AutoAWQ default.")
    p.add_argument("--max-chunk-memory", type=int, default=1024 * 1024 * 1024,
                   help="activation-statistics chunk budget in bytes (default 1GB)")
    args = p.parse_args()

    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[autoawq] loading {args.model} (dtype={args.dtype})")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }
    print(f"[autoawq] quant_config = {quant_config}")
    print(f"[autoawq] calibration: max_samples={args.max_calib_samples} "
          f"seq_len={args.max_calib_seq_len}")

    quant_kwargs = dict(
        quant_config=quant_config,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.max_calib_seq_len,
        max_chunk_memory=args.max_chunk_memory,
    )
    if args.n_parallel_calib_samples is not None:
        quant_kwargs["n_parallel_calib_samples"] = args.n_parallel_calib_samples
    model.quantize(tokenizer, **quant_kwargs)

    os.makedirs(args.out, exist_ok=True)
    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[autoawq] wrote {args.out}")


if __name__ == "__main__":
    main()
