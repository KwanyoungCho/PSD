"""Try multiple Int4 variants to find a working one on SM 86."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int4WeightOnlyConfig
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


N, in_f, out_f = 8, 4096, 11008
torch.manual_seed(0)
x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02

dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
with torch.no_grad(): dense.weight.copy_(W)
t_dense = bench_fn(lambda: F.linear(x, dense.weight))
print(f"dense bf16: {t_dense:.3f} ms")

for fmt in Int4PackingFormat:
    for gs in [32, 64, 128]:
        for ver in [1, 2]:
            try:
                q = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
                with torch.no_grad(): q.weight.copy_(W)
                quantize_(q, Int4WeightOnlyConfig(group_size=gs, int4_packing_format=fmt, version=ver))
                t = bench_fn(lambda: F.linear(x, q.weight))
                print(f"  {str(fmt):<55s} gs={gs} v{ver}: {t:.3f} ms  ({t/t_dense:.2f}x)")
            except Exception as e:
                msg = str(e)[:80]
                print(f"  {str(fmt):<55s} gs={gs} v{ver}: FAIL: {msg}")
