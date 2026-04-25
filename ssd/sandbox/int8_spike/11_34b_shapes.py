"""Microbench on 34B TP=4 shapes."""
import torch, torch.nn as nn, torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig


def bench(fn, n=50, w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n


# 34B TP=4 shapes (hidden=8192, head=128, num_heads=64, num_kv=8)
shapes = [
    (2,   8192, 2560, "AR decode 34B qkv"),
    (2,   8192, 11008, "AR decode 34B gate_up"),
    (2,   5504, 8192, "AR decode 34B down_proj"),
    (8,   8192, 2560, "verify 8 34B qkv"),
    (8,   8192, 11008, "verify 8 34B gate_up"),
    (8,   5504, 8192, "verify 8 34B down_proj"),
    (8,   2048, 8192, "verify 8 34B o_proj (row)"),
    (338, 8192, 2560, "prefill 338 34B qkv"),
    (338, 8192, 11008, "prefill 338 34B gate_up"),
    (338, 5504, 8192, "prefill 338 34B down_proj"),
]

print(f"{'shape':<35s} {'dense':>8s} {'int8':>14s} {'int4':>14s}")
for N, in_f, out_f, label in shapes:
    torch.manual_seed(0)
    x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02

    dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): dense.weight.copy_(W)
    t_d = bench(lambda: F.linear(x, dense.weight))

    q8 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q8.weight.copy_(W)
    quantize_(q8, Int8WeightOnlyConfig())
    t_8 = bench(lambda: F.linear(x, q8.weight))

    try:
        q4 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
        with torch.no_grad(): q4.weight.copy_(W)
        quantize_(q4, Int4WeightOnlyConfig(group_size=128))
        t_4 = bench(lambda: F.linear(x, q4.weight))
        s4 = f"{t_4:.3f}({t_4/t_d:.2f}x)"
    except Exception as e:
        s4 = f"FAIL {str(e)[:25]}"

    print(f"{label:<35s} {t_d:>6.3f}ms {t_8:>8.3f}ms({t_8/t_d:.2f}x) {s4:>14s}")
