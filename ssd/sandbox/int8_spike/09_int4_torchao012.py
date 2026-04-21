"""Check if torchao 0.12 (original ssd env) supports int4 tile_packed path."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int4WeightOnlyConfig


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


print("torchao config signature:")
import inspect
print(inspect.signature(Int4WeightOnlyConfig))

# Torchao 0.12 Int4WeightOnlyConfig options are different
# Let's see what we can do

N, in_f, out_f = 8, 4096, 11008
torch.manual_seed(0)
x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02

dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
with torch.no_grad(): dense.weight.copy_(W)
t_dense = bench_fn(lambda: F.linear(x, dense.weight))
print(f"dense bf16: {t_dense:.3f} ms")

# Try default int4 in 0.12
try:
    q = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q.weight.copy_(W)
    cfg = Int4WeightOnlyConfig(group_size=128)
    quantize_(q, cfg)
    t = bench_fn(lambda: F.linear(x, q.weight))
    print(f"int4 default (gs=128): {t:.3f} ms  ({t/t_dense:.2f}x)")
    print(f"  weight type: {type(q.weight.data).__name__}")
except Exception as e:
    print(f"int4 default: FAIL {e}")

# Also try with layout override if available
try:
    from torchao.dtypes import TensorCoreTiledLayout
    q = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q.weight.copy_(W)
    cfg = Int4WeightOnlyConfig(group_size=128, layout=TensorCoreTiledLayout())
    quantize_(q, cfg)
    t = bench_fn(lambda: F.linear(x, q.weight))
    print(f"int4 TensorCoreTiled: {t:.3f} ms  ({t/t_dense:.2f}x)")
except Exception as e:
    print(f"int4 TensorCoreTiled: FAIL {e}")
