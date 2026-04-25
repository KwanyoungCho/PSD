"""Phase 0 smoke test: does sgl-kernel's Marlin W4A16 path run on this box?

Goals:
  - confirm `awq_marlin_repack` + `gptq_marlin_gemm` actually execute
  - confirm fp16 and bf16 activation support
  - confirm decode-sized (M=1..8) shapes work
  - confirm CUDA graph capture works
  - sanity-check numerical agreement with a reference dequant path

Run: python -O sandbox/awq_spike/00_marlin_smoke.py
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("SSD_CUDA_ARCH", "8.6"))

import torch
from sgl_kernel import awq_marlin_repack, gptq_marlin_gemm, awq_dequantize
from sgl_kernel.scalar_type import scalar_types


def make_fake_awq_weight(
    in_features: int,
    out_features: int,
    *,
    group_size: int = 128,
    bits: int = 4,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 0,
):
    """Produce a synthetic AWQ-format quantized linear weight + scales + zeros.

    AutoAWQ packs 8 int4 values per int32 along the *input* dim. Layout:
      qweight: [in_features, out_features // (32/bits)] int32
      qzeros:  [in_features // group_size, out_features // (32/bits)] int32
      scales:  [in_features // group_size, out_features] fp16/bf16
    """
    torch.manual_seed(seed)
    assert in_features % group_size == 0
    pack_factor = 32 // bits
    assert out_features % pack_factor == 0

    num_groups = in_features // group_size
    # Start from float weight, fake-quantize to int4 [0, 15] per-group.
    w_fp = torch.randn(out_features, in_features, device=device, dtype=torch.float32) * 0.02

    w_grouped = w_fp.view(out_features, num_groups, group_size)
    # Per-group symmetric-ish scale with zero-point around 8 (int4 midpoint).
    w_max = w_grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale_per_group = (2.0 * w_max / (2 ** bits - 1)).squeeze(-1)  # [out, num_groups]
    # Quantize per-group, zero-point = 8 (shift to unsigned)
    w_q = (w_grouped / scale_per_group.unsqueeze(-1)).round().to(torch.int32) + 8
    w_q = w_q.clamp(0, 2 ** bits - 1).view(out_features, in_features).contiguous()

    # AWQ layout: rows = in_features, cols = out_features / pack_factor
    qweight = torch.zeros(in_features, out_features // pack_factor, dtype=torch.int32, device=device)
    # AutoAWQ interleave order [0, 2, 4, 6, 1, 3, 5, 7] for 4-bit per column group
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)
    for col in range(out_features // pack_factor):
        packed = torch.zeros(in_features, dtype=torch.int32, device=device)
        for i, bit_shift_idx in enumerate(order):
            val = w_q[col * pack_factor + bit_shift_idx]  # [in_features]
            packed |= (val.to(torch.int32) & 0xF) << (i * 4)
        qweight[:, col] = packed

    qzeros = torch.full(
        (num_groups, out_features // pack_factor),
        fill_value=0,
        dtype=torch.int32,
        device=device,
    )
    for col in range(out_features // pack_factor):
        packed = torch.zeros(num_groups, dtype=torch.int32, device=device)
        for i, bit_shift_idx in enumerate(order):
            val = torch.full((num_groups,), 8, dtype=torch.int32, device=device)
            packed |= (val & 0xF) << (i * 4)
        qzeros[:, col] = packed

    scales = scale_per_group.T.contiguous().to(dtype)  # [num_groups, out_features]
    return qweight, qzeros, scales


def run_marlin_once(
    M: int, K: int, N: int,
    *, group_size: int = 128, dtype: torch.dtype = torch.float16, capture_graph: bool = False,
):
    torch.cuda.empty_cache()
    qweight, qzeros, scales = make_fake_awq_weight(K, N, group_size=group_size, dtype=dtype)

    # Repack AWQ → Marlin
    marlin_qweight = awq_marlin_repack(qweight, size_k=K, size_n=N, num_bits=4)

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1

    workspace = torch.zeros(N // 64 * 16, dtype=torch.int32, device="cuda")

    def _call():
        return gptq_marlin_gemm(
            a=x,
            c=None,
            b_q_weight=marlin_qweight,
            b_scales=scales,
            global_scale=None,
            b_zeros=qzeros,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=scalar_types.uint4,          # AWQ: uint4 (zero-point packed, not uint4b8)
            size_m=M,
            size_n=N,
            size_k=K,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    # Warmup
    y = _call()
    torch.cuda.synchronize()

    if capture_graph:
        static_x = torch.empty_like(x)
        static_y = torch.empty_like(y)

        def _graph_call():
            static_y.copy_(
                gptq_marlin_gemm(
                    a=static_x, c=None,
                    b_q_weight=marlin_qweight, b_scales=scales, global_scale=None,
                    b_zeros=qzeros, g_idx=None, perm=None, workspace=workspace,
                    b_q_type=scalar_types.uint4, size_m=M, size_n=N, size_k=K,
                    is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
                    is_zp_float=False,
                )
            )

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            static_x.copy_(x)
            _graph_call()  # warm
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _graph_call()
        static_x.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        y = static_y

    # Reference: awq_dequantize → dense matmul
    w_ref = awq_dequantize(qweight, scales, qzeros)   # [K, N] in activation dtype
    y_ref = x @ w_ref

    err = (y - y_ref).abs().max().item()
    rel = err / (y_ref.abs().max().item() + 1e-6)
    return y.shape, err, rel


def main():
    cap = torch.cuda.get_device_capability()
    dev = torch.cuda.get_device_name()
    print(f"device: {dev}  sm_{cap[0]}{cap[1]}")

    shapes = [
        # decode/verify-sized (Llama-3 8B per-rank tp=2: hidden=4096/1, ffn=14336/2=7168)
        (1, 4096, 4096),
        (1, 4096, 12288),    # qkv combined
        (1, 4096, 28672),    # gate_up combined
        (1, 14336, 4096),    # down_proj
        (4, 4096, 4096),
        (8, 4096, 4096),
        # prefill-like
        (256, 4096, 4096),
        (1024, 4096, 4096),
    ]

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\n=== dtype={dtype} ===")
        for (M, K, N) in shapes:
            try:
                shape, err, rel = run_marlin_once(M, K, N, dtype=dtype, capture_graph=(M == 1))
                print(f"  M={M:4d} K={K:5d} N={N:5d}  y={shape}  max_err={err:.4f}  rel={rel:.4f}")
            except Exception as e:
                print(f"  M={M:4d} K={K:5d} N={N:5d}  FAILED: {type(e).__name__}: {e}")

    print("\ndone.")


if __name__ == "__main__":
    main()
