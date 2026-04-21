"""Microbench: compare dense bf16 vs int8 weight-only on SSD-sized matmuls.

Profile where time goes: dequant vs matmul.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig


def bench_fn(fn, n_iter=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


shapes = [
    # (batch, in, out, label)
    (338, 4096, 11008, "gate_up 8B prefill"),
    (338, 5504, 4096, "down_proj 8B prefill TP=2"),
    (8,   4096, 4096, "o_proj 8B decode TP=2 (verify 8 tok)"),
    (8,   4096, 11008, "gate_up 8B verify"),
    (338, 4096, 128256, "lm_head 8B prefill"),
    (2,   4096, 11008, "gate_up 8B AR decode"),
]

for N, in_f, out_f, label in shapes:
    print(f"\n=== {label}  [{N} x {in_f}] @ [{in_f} x {out_f}] ===")
    torch.manual_seed(0)
    x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = None

    # dense bf16 baseline
    dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dense.weight.copy_(W)
    t_dense = bench_fn(lambda: F.linear(x, dense.weight, bias))

    # int8 weight only via torchao
    q = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        q.weight.copy_(W)
    quantize_(q, Int8WeightOnlyConfig())
    t_int8 = bench_fn(lambda: F.linear(x, q.weight, bias))

    print(f"  dense bf16: {t_dense:.3f} ms")
    print(f"  int8 wo   : {t_int8:.3f} ms  ({t_int8/t_dense:.2f}x of dense)")

    # Also benchmark "manual dequant + matmul" to see if it matches int8_wo path
    aqt = q.weight.data
    if hasattr(aqt, 'tensor_impl'):
        scale = aqt.tensor_impl.scale
        int_data = aqt.tensor_impl.int_data
        def manual():
            w_fp = int_data.to(x.dtype) * scale.unsqueeze(1).to(x.dtype)
            return F.linear(x, w_fp, bias)
        t_manual = bench_fn(manual)
        print(f"  manual dq : {t_manual:.3f} ms  ({t_manual/t_dense:.2f}x of dense) [dequant + bf16 matmul]")
