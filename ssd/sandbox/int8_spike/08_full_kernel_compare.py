"""Compare all feasible kernel paths: dense bf16, int8 WO, int4 WO tile_packed, W8A8 dyn."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import (
    quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat


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
    (2,   4096, 11008, "AR decode 1-2 tok  (gate_up)"),
    (8,   4096, 6144,  "verify 8 tok       (qkv)"),
    (8,   4096, 11008, "verify 8 tok       (gate_up)"),
    (8,   5504, 4096,  "verify 8 tok       (down_proj)"),
    (338, 4096, 6144,  "prefill 338 tok    (qkv)"),
    (338, 4096, 11008, "prefill 338 tok    (gate_up)"),
    (338, 5504, 4096,  "prefill 338 tok    (down_proj)"),
]


print(f"{'shape':<45s} {'dense':>8s} {'int8 WO':>10s} {'int4 WO':>10s} {'int8 dyn':>10s}")

for N, in_f, out_f, label in shapes:
    torch.manual_seed(0)
    x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02

    dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): dense.weight.copy_(W)
    t_dense = bench_fn(lambda: F.linear(x, dense.weight))

    q8 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q8.weight.copy_(W)
    quantize_(q8, Int8WeightOnlyConfig())
    t_q8 = bench_fn(lambda: F.linear(x, q8.weight))

    q4 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q4.weight.copy_(W)
    try:
        quantize_(q4, Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            version=2,
        ))
        t_q4 = bench_fn(lambda: F.linear(x, q4.weight))
        s4 = f"{t_q4:.2f}ms({t_q4/t_dense:.2f}x)"
    except Exception as e:
        s4 = "FAIL"

    qdyn = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): qdyn.weight.copy_(W)
    try:
        quantize_(qdyn, Int8DynamicActivationInt8WeightConfig())
        t_qdyn = bench_fn(lambda: F.linear(x, qdyn.weight))
        sdyn = f"{t_qdyn:.2f}ms({t_qdyn/t_dense:.2f}x)"
    except Exception as e:
        sdyn = f"FAIL {str(e)[:30]}"

    s_dense = f"{t_dense:.2f}ms"
    s8 = f"{t_q8:.2f}ms({t_q8/t_dense:.2f}x)"
    print(f"{label:<45s} {s_dense:>8s} {s8:>15s} {s4:>15s} {sdyn:>18s}")
