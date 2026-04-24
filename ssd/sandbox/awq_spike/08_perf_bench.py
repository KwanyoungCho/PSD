"""Phase 7 micro + end-to-end bench comparing dense vs AWQ Marlin.

A) Micro: local TP-linear matmul, decode-M sweep, dense F.linear vs
   `awq_matmul(AwqQuantState)` on the same logical weights.
B) E2E AR decode tok/s for layerskip-llama3-8B TP=2 with and without AWQ.

Run (needs 2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/08_perf_bench.py
"""
import os
import time

os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13250")


import torch

from ssd.layers.linear import ColumnParallelLinear, RowParallelLinear
from ssd.quant.init_context import quant_init_context
from ssd.quant.pack import rtn_quantize_w4a16
from ssd.quant.build import RawAwqTensors, build_awq_state


# ---------- (A) microbench ----------

def _bench_matmul(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6   # microseconds


def microbench():
    print("\n[A] local-matmul microbench  (μs/call, lower=better)")
    print(f"   {'shape':<28s} {'dense':>10s} {'awq_marlin':>12s} {'speedup':>9s}")

    dtype = torch.bfloat16
    shapes = [
        ("qkv_proj tp2 dec M=1",    1,  4096, 6144 // 2),   # 3072
        ("qkv_proj tp2 verify M=8", 8,  4096, 3072),
        ("gate_up  tp2 dec M=1",    1,  4096, 14336),       # 2 × 14336/2
        ("gate_up  tp2 verify M=8", 8,  4096, 14336),
        ("down_proj tp2 dec M=1",   1,  14336 // 2, 4096),  # RowParallel
        ("o_proj tp2 dec M=1",      1,  4096 // 2, 4096),
        ("prefill qkv M=256",       256, 4096, 3072),
        ("prefill gate_up M=256",   256, 4096, 14336),
    ]

    for label, M, K, N in shapes:
        torch.manual_seed(0)
        w = (torch.randn(N, K, device="cuda", dtype=dtype) * 0.02).contiguous()
        x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1

        dense_fn = lambda: torch.nn.functional.linear(x, w)

        # Build quant module
        with quant_init_context():
            mod = ColumnParallelLinear(K, N, bias=False, tp_size=1)
        qw, qz, sc, _ = rtn_quantize_w4a16(w, group_size=128)
        raw = RawAwqTensors(qweight=qw.cpu(), qzeros=qz.cpu(), scales=sc.cpu(),
                            in_features=K, out_features=N, group_size=128)
        state = build_awq_state(raw, device=torch.device("cuda"))
        mod.attach_quant_state(state)
        quant_fn = lambda: mod(x)

        us_dense = _bench_matmul(dense_fn)
        us_quant = _bench_matmul(quant_fn)
        speedup = us_dense / us_quant
        print(f"   {label:<28s} {us_dense:>10.1f} {us_quant:>12.1f} {speedup:>8.2f}x")


# ---------- (B) E2E ----------

MODEL = "/data2/chokwans99/models/layerskip-llama3-8B"
ARTIFACT = "/tmp/awq_artifacts/layerskip8b_tp2"


def _end_to_end_runner(enable_awq: bool, label: str):
    print(f"\n[B] E2E TP=2  variant={label}")
    from ssd import LLM, SamplingParams
    kwargs = dict(
        model=MODEL,
        num_gpus=2,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.4,
        enforce_eager=False,
    )
    if enable_awq:
        kwargs.update(
            target_quant_enabled=True,
            target_quant_backend="awq_marlin",
            target_quant_awq_artifact=ARTIFACT,
        )
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_new_tokens=128)

    t0 = time.perf_counter()
    out = llm.generate(["The capital of France is"], sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    # generate() returns (list_of_dicts, METRICS_dict)
    outputs = out[0] if isinstance(out, tuple) else out
    ntok = len(outputs[0]["token_ids"])
    print(f"   {label} tok={ntok} time={dt:.2f}s throughput={ntok/dt:.1f} tok/s")
    del llm
    torch.cuda.empty_cache()


def e2e(variant: str):
    """Run only ONE variant per process — spawn contexts don't recycle cleanly."""
    if variant == "dense":
        _end_to_end_runner(False, "dense-bf16")
    elif variant == "awq":
        _end_to_end_runner(True, "awq-marlin")
    else:
        raise ValueError(variant)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "micro"
    if mode == "micro":
        microbench()
    elif mode in ("dense", "awq"):
        e2e(mode)
    else:
        print(f"usage: {sys.argv[0]} [micro|dense|awq]")
