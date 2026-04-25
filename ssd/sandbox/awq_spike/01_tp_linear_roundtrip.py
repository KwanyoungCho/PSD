"""Round-trip test for Phase 2 TP-linear quant path.

Plan:
  1. Generate a dense fp16/bf16 weight.
  2. RTN-quantize it → AWQ-format tensors.
  3. Run a reference dense matmul using the dequantized weight.
  4. Build AwqQuantState + attach to a ColumnParallelLinear.
  5. Run quant forward.
  6. Compare outputs.

This isolates Phase 2's TP linear integration from any model/loader wiring.

Run: CUDA_VISIBLE_DEVICES=0 python -O sandbox/awq_spike/01_tp_linear_roundtrip.py
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("SSD_CUDA_ARCH", "8.6"))

import torch

from ssd.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from ssd.quant.init_context import quant_init_context
from ssd.quant.pack import rtn_quantize_w4a16
from ssd.quant.build import RawAwqTensors, build_awq_state


def _quant_col_module(in_f, out_f, dtype):
    # In quant mode the `weight` is a meta placeholder — do NOT call .cuda()
    # (it would try to move the meta tensor and fail). The module's real GPU
    # state is the AwqQuantState attached later. This mirrors the real flow
    # where torch.set_default_device("cuda") ensures tensors default to CUDA
    # and meta weights are never realized.
    with quant_init_context():
        mod = ColumnParallelLinear(in_f, out_f, bias=False, tp_size=1)
    assert mod.weight.device.type == "meta", \
        f"quant-mode weight should be on meta, got {mod.weight.device}"
    return mod


def _quant_row_module(in_f, out_f, dtype):
    with quant_init_context():
        mod = RowParallelLinear(in_f, out_f, bias=False, tp_size=1)
    assert mod.weight.device.type == "meta"
    return mod


def _attach_rtn(mod, weight_dense, group_size=128):
    """RTN-quantize a dense weight and attach to a TP linear module."""
    qw, qz, sc, dq = rtn_quantize_w4a16(weight_dense, group_size=group_size)
    raw = RawAwqTensors(
        qweight=qw.cpu(), qzeros=qz.cpu(), scales=sc.cpu(),
        in_features=weight_dense.shape[1],
        out_features=weight_dense.shape[0],
        group_size=group_size,
    )
    state = build_awq_state(raw, device=torch.device("cuda"))
    mod.attach_quant_state(state)
    return dq.cuda()   # dequantized reference weight (same precision as Marlin uses internally)


def test_column_parallel(in_f=4096, out_f=4096, M=4, dtype=torch.float16):
    print(f"\n[col-parallel]  in={in_f} out={out_f} M={M} dtype={dtype}")
    torch.manual_seed(42)
    w = (torch.randn(out_f, in_f, device="cuda", dtype=dtype) * 0.02).contiguous()
    x = torch.randn(M, in_f, device="cuda", dtype=dtype) * 0.1

    # Dense reference (true, not dequant)
    y_true = torch.nn.functional.linear(x, w)

    mod = _quant_col_module(in_f, out_f, dtype)
    w_dq = _attach_rtn(mod, w)
    # Reference using the dequantized weight (what Marlin will reproduce)
    y_ref = torch.nn.functional.linear(x, w_dq)
    y_q = mod(x)

    err_true = (y_q - y_true).abs().max().item() / (y_true.abs().max().item() + 1e-6)
    err_ref = (y_q - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"  max-rel err vs dense(true)        = {err_true:.4f}")
    print(f"  max-rel err vs dense(dequant ref) = {err_ref:.4f}")
    assert err_ref < 0.02, f"Marlin vs dequant ref mismatch too large: {err_ref}"


def test_row_parallel(in_f=14336, out_f=4096, M=4, dtype=torch.float16):
    print(f"\n[row-parallel]  in={in_f} out={out_f} M={M} dtype={dtype}")
    torch.manual_seed(43)
    w = (torch.randn(out_f, in_f, device="cuda", dtype=dtype) * 0.02).contiguous()
    x = torch.randn(M, in_f, device="cuda", dtype=dtype) * 0.1

    y_true = torch.nn.functional.linear(x, w)

    mod = _quant_row_module(in_f, out_f, dtype)
    w_dq = _attach_rtn(mod, w)
    y_ref = torch.nn.functional.linear(x, w_dq)
    y_q = mod(x)

    err_true = (y_q - y_true).abs().max().item() / (y_true.abs().max().item() + 1e-6)
    err_ref = (y_q - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"  max-rel err vs dense(true)        = {err_true:.4f}")
    print(f"  max-rel err vs dense(dequant ref) = {err_ref:.4f}")
    assert err_ref < 0.02, f"Marlin vs dequant ref mismatch too large: {err_ref}"


def test_decode_shapes(dtype=torch.float16):
    print(f"\n[decode shapes]  dtype={dtype}")
    for (in_f, out_f, M) in [
        (4096, 4096, 1),
        (4096, 12288, 1),    # Llama3-8B qkv
        (4096, 28672, 1),    # Llama3-8B gate_up
        (14336, 4096, 1),    # Llama3-8B down_proj
        (4096, 4096, 8),     # verify-like
    ]:
        torch.manual_seed(in_f + out_f + M)
        w = (torch.randn(out_f, in_f, device="cuda", dtype=dtype) * 0.02).contiguous()
        x = torch.randn(M, in_f, device="cuda", dtype=dtype) * 0.1

        mod = _quant_col_module(in_f, out_f, dtype)
        w_dq = _attach_rtn(mod, w)
        y_ref = torch.nn.functional.linear(x, w_dq)
        y_q = mod(x)
        err = (y_q - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-6)
        print(f"  in={in_f:5d} out={out_f:5d} M={M:2d}  rel_err={err:.4f}")
        assert err < 0.02, f"shape {in_f}x{out_f} M={M} rel_err={err} too large"


def test_graph_safety(dtype=torch.float16):
    print(f"\n[graph safety]  dtype={dtype}")
    torch.manual_seed(44)
    in_f, out_f, M = 4096, 4096, 1
    w = (torch.randn(out_f, in_f, device="cuda", dtype=dtype) * 0.02).contiguous()

    mod = _quant_col_module(in_f, out_f, dtype)
    w_dq = _attach_rtn(mod, w)

    static_x = torch.randn(M, in_f, device="cuda", dtype=dtype) * 0.1
    # Warmup on a side stream
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = mod(static_x)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    y_buf = torch.empty(M, out_f, device="cuda", dtype=dtype)
    with torch.cuda.graph(g):
        y_buf.copy_(mod(static_x))

    # Replay with new activation values, dense-equivalent result should match.
    static_x.copy_(torch.randn(M, in_f, device="cuda", dtype=dtype) * 0.1)
    g.replay()
    torch.cuda.synchronize()
    y_ref = torch.nn.functional.linear(static_x, w_dq)
    err = (y_buf - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"  graph replay rel_err = {err:.4f}")
    assert err < 0.02, f"graph replay error too large: {err}"


def main():
    for dt in [torch.float16, torch.bfloat16]:
        test_column_parallel(dtype=dt)
        test_row_parallel(dtype=dt)
        test_decode_shapes(dtype=dt)
        test_graph_safety(dtype=dt)
    print("\nALL OK.")


if __name__ == "__main__":
    main()
