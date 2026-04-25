"""Diagnose AWQ layout mismatch.

Sanity ladder:
  A. pack → unpack round-trip for fake int4 matrix.
  B. manual dequant via our unpack vs sgl_kernel.awq_dequantize (must agree).
  C. manual dequant matmul vs gptq_marlin_gemm (must agree if Marlin
     consumes our layout directly).

Run WITHOUT -O so asserts fire.
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("SSD_CUDA_ARCH", "8.6"))

import torch

from sgl_kernel import awq_dequantize, gptq_marlin_gemm, awq_marlin_repack
from sgl_kernel.scalar_type import scalar_types

from ssd.quant.pack import awq_pack_4bit, awq_unpack_4bit, rtn_quantize_w4a16


def step_a_pack_roundtrip():
    print("[A] pack → unpack round-trip")
    torch.manual_seed(0)
    rows, cols = 16, 32       # cols must be /8
    values = torch.randint(0, 16, (rows, cols), dtype=torch.int32, device="cuda")
    packed = awq_pack_4bit(values, out_dim=1)
    assert packed.shape == (rows, cols // 8), packed.shape
    recovered = awq_unpack_4bit(packed, pack_dim=-1)
    assert recovered.shape == values.shape, recovered.shape
    assert torch.equal(recovered, values), "pack/unpack not inverse"
    print("   OK")


def _manual_dequant(qweight, qzeros, scales, *, group_size, dtype):
    """Dequantize AWQ (qweight, qzeros, scales) to dense [K, N] via our own unpack.

    Returned tensor is [K, N] in `dtype` (matches awq_dequantize layout).
    """
    # qweight: [K, N // 8] → [K, N] int
    w_q = awq_unpack_4bit(qweight, pack_dim=-1)          # [K, N] int32
    # qzeros: [num_groups, N // 8] → [num_groups, N]
    z_q = awq_unpack_4bit(qzeros, pack_dim=-1)           # [num_groups, N] int32
    K, N = w_q.shape
    num_groups = K // group_size

    # Broadcast scales/zeros along the group dim
    scales_e = scales.to(torch.float32).repeat_interleave(group_size, dim=0)   # [K, N]
    zeros_e = z_q.to(torch.float32).repeat_interleave(group_size, dim=0)       # [K, N]
    dequant = (w_q.to(torch.float32) - zeros_e) * scales_e
    return dequant.to(dtype)


def step_b_dequant_agreement(dtype=torch.float16):
    print(f"\n[B] manual dequant vs awq_dequantize   dtype={dtype}")
    torch.manual_seed(1)
    K, N, group_size = 128, 64, 64   # small but valid (K % group == 0, N % 8 == 0)
    dense = torch.randn(N, K, device="cuda", dtype=dtype) * 0.05
    qw, qz, sc, dq_from_rtn = rtn_quantize_w4a16(dense, group_size=group_size)
    qw_cuda = qw.to("cuda").contiguous()
    qz_cuda = qz.to("cuda").contiguous()
    sc_cuda = sc.to("cuda").contiguous()

    # sgl-kernel's own dequantizer
    dq_kernel = awq_dequantize(qw_cuda, sc_cuda, qz_cuda)   # [K, N]
    # Our manual unpack+dequant
    dq_manual = _manual_dequant(qw_cuda, qz_cuda, sc_cuda, group_size=group_size, dtype=dtype)
    # RTN's own returned dequant (shape [N, K])
    dq_rtn = dq_from_rtn.to("cuda").T.contiguous()   # [K, N]

    err_km = (dq_kernel - dq_manual).abs().max().item()
    err_kr = (dq_kernel - dq_rtn).abs().max().item()
    err_mr = (dq_manual - dq_rtn).abs().max().item()
    print(f"   kernel vs manual  : max |Δ| = {err_km:.6f}")
    print(f"   kernel vs rtn-dq  : max |Δ| = {err_kr:.6f}")
    print(f"   manual vs rtn-dq  : max |Δ| = {err_mr:.6f}")
    # Expect numerical equality up to fp16 roundoff
    rel = (dq_kernel.float() - dq_rtn.float()).abs().max().item() / (dq_rtn.abs().max().item() + 1e-6)
    print(f"   kernel vs rtn-dq  : rel = {rel:.4f}")


def step_c_marlin_agreement(dtype=torch.float16):
    print(f"\n[C] Marlin(qweight, scales, qzeros) vs manual dequant matmul  dtype={dtype}")
    torch.manual_seed(2)
    M, K, N = 4, 4096, 4096
    group_size = 128
    dense = torch.randn(N, K, device="cuda", dtype=dtype) * 0.02   # [N, K]
    qw, qz, sc, dq_from_rtn = rtn_quantize_w4a16(dense, group_size=group_size)
    qw_cuda = qw.to("cuda").contiguous()
    qz_cuda = qz.to("cuda").contiguous()
    sc_cuda = sc.to("cuda").contiguous()

    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1

    # Reference 1: dense matmul with the dequant weight from RTN
    w_ref = dq_from_rtn.to("cuda").contiguous()   # [N, K]
    y_ref = x @ w_ref.T                            # [M, N]

    # Reference 2: kernel dequant + matmul
    w_kernel_dq = awq_dequantize(qw_cuda, sc_cuda, qz_cuda)   # [K, N]
    y_kernel_ref = x @ w_kernel_dq

    # Marlin path: repack qweight, try qzeros raw first
    marlin_qw = awq_marlin_repack(qw_cuda, size_k=K, size_n=N, num_bits=4)
    workspace = torch.zeros(N // 64 * 16, dtype=torch.int32, device="cuda")

    try:
        y_marlin = gptq_marlin_gemm(
            a=x, c=None, b_q_weight=marlin_qw,
            b_scales=sc_cuda, global_scale=None, b_zeros=qz_cuda,
            g_idx=None, perm=None, workspace=workspace,
            b_q_type=scalar_types.uint4,
            size_m=M, size_n=N, size_k=K, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )
        err_ref = (y_marlin - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-6)
        err_kernel = (y_marlin - y_kernel_ref).abs().max().item() / (y_kernel_ref.abs().max().item() + 1e-6)
        print(f"   marlin vs dense(rtn-dq)     rel = {err_ref:.4f}")
        print(f"   marlin vs dense(kernel-dq)  rel = {err_kernel:.4f}")
    except Exception as e:
        print(f"   Marlin call failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    step_a_pack_roundtrip()
    step_b_dequant_agreement()
    step_c_marlin_agreement()
