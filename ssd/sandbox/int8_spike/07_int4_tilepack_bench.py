"""INT4 TILE_PACKED_TO_4D v2 (tinygemm) across SSD-relevant shapes."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
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
    (338, 4096, 11008, "gate_up prefill 8B"),
    (338, 5504, 4096, "down_proj prefill 8B TP=2"),
    (8,   4096, 11008, "gate_up verify 8 tok"),
    (8,   5504, 4096, "down_proj verify 8 tok TP=2"),
    (2,   4096, 11008, "gate_up AR decode"),
    (338, 4096, 128256, "lm_head prefill"),
    (338, 4096, 6144,  "qkv prefill 8B TP=2 (32Q+8KV+8KV)*128/2"),
    (8,   4096, 6144,  "qkv verify 8 tok TP=2"),
]

for N, in_f, out_f, label in shapes:
    print(f"\n=== {label}  [{N}x{in_f}] @ [{in_f}x{out_f}] ===")
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

    q4 = None
    t_q4 = None
    try:
        q4 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
        with torch.no_grad(): q4.weight.copy_(W)
        quantize_(q4, Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            version=2,
        ))
        t_q4 = bench_fn(lambda: F.linear(x, q4.weight))
    except Exception as e:
        print(f"  int4 tile4d fail: {e}")

    print(f"  dense bf16          : {t_dense:.3f} ms  (1.00x)")
    print(f"  int8 wo             : {t_q8:.3f} ms  ({t_q8/t_dense:.2f}x)")
    if t_q4 is not None:
        print(f"  int4 tile_packed_4d : {t_q4:.3f} ms  ({t_q4/t_dense:.2f}x of dense) <-- fast path")

# Also test numerical correctness of Int4 tile_packed path
print("\n=== Numerical sanity (gate_up) ===")
torch.manual_seed(0)
N, in_f, out_f = 338, 4096, 11008
x = torch.randn(N, in_f, device="cuda", dtype=torch.bfloat16)
W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02
dense = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
with torch.no_grad(): dense.weight.copy_(W)
y_dense = F.linear(x, dense.weight)

q4 = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
with torch.no_grad(): q4.weight.copy_(W)
quantize_(q4, Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
    version=2,
))
y_q4 = F.linear(x, q4.weight)
diff = (y_q4.float() - y_dense.float()).abs().max().item()
cos = F.cosine_similarity(y_q4.float().flatten(), y_dense.float().flatten(), dim=0).item()
print(f"  max abs diff vs dense: {diff:.4g}  cosine: {cos:.6f}")
