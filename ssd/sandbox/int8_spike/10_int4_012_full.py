"""Full microbench: Int4 default (TensorCoreTiled) in torchao 0.12 across SSD shapes."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig


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
    (2,   4096, 11008, "AR decode         (gate_up)"),
    (8,   4096, 6144,  "verify 8 tok      (qkv TP=2)"),
    (8,   4096, 11008, "verify 8 tok      (gate_up)"),
    (8,   5504, 4096,  "verify 8 tok      (down_proj TP=2)"),
    (338, 4096, 6144,  "prefill 338 tok   (qkv TP=2)"),
    (338, 4096, 11008, "prefill 338 tok   (gate_up)"),
    (338, 5504, 4096,  "prefill 338 tok   (down_proj TP=2)"),
    (338, 4096, 64128, "prefill lm_head TP=2 (128256/2)"),
    (2,   4096, 64128, "AR lm_head TP=2"),
]

print(f"{'shape':<45s} {'dense':>10s} {'int8':>15s} {'int4':>15s}")
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

    try:
        q4 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
        with torch.no_grad(): q4.weight.copy_(W)
        quantize_(q4, Int4WeightOnlyConfig(group_size=128))
        t_q4 = bench_fn(lambda: F.linear(x, q4.weight))
        s4 = f"{t_q4:.3f}ms ({t_q4/t_dense:.2f}x)"
    except Exception as e:
        s4 = f"FAIL {str(e)[:40]}"

    s_dense = f"{t_dense:.3f}ms"
    s8 = f"{t_q8:.3f}ms ({t_q8/t_dense:.2f}x)"
    print(f"{label:<45s} {s_dense:>10s} {s8:>18s} {s4:>20s}")

# Numerical sanity
print("\n=== Numerical sanity (cosine vs dense) ===")
for label, N, in_f, out_f in [("gate_up", 8, 4096, 11008), ("down_proj", 8, 5504, 4096)]:
    torch.manual_seed(0)
    x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02
    dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): dense.weight.copy_(W)
    y_d = F.linear(x, dense.weight)

    q8 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q8.weight.copy_(W)
    quantize_(q8, Int8WeightOnlyConfig())
    y_8 = F.linear(x, q8.weight)
    c8 = F.cosine_similarity(y_d.float().flatten(), y_8.float().flatten(), dim=0).item()

    q4 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad(): q4.weight.copy_(W)
    quantize_(q4, Int4WeightOnlyConfig(group_size=128))
    y_4 = F.linear(x, q4.weight)
    c4 = F.cosine_similarity(y_d.float().flatten(), y_4.float().flatten(), dim=0).item()

    print(f"  {label}: int8 cos={c8:.6f}  int4 cos={c4:.6f}")
