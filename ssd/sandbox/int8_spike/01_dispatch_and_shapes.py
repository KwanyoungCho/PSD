"""Phase 0 sandbox — checks 1, 2, 3.

1. quantize_() walker targets nn.Linear only (SSD custom is not Linear).
2. Try storage contracts (A)-(D) for moving a quantized tensor onto a module
   that isn't nn.Linear, and see which of them keep F.linear dispatching to
   the int8 kernel.
3. Same test at SSD-like local shard sizes + packed (qkv, gate_up).

Usage:
  python sandbox/int8_spike/01_dispatch_and_shapes.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig


def header(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# --- Custom linear: not nn.Linear, mimics SSD's ColumnParallelLinear/etc. -----
class SSDLikeLinear(nn.Module):
    """Mirror of SSD's custom TP linear: not an nn.Linear, just holds a weight."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def is_affine_quant_tensor(t):
    return type(t).__name__ != "Tensor" and type(t).__name__ != "Parameter"


def check_dispatch(module, x, label):
    y = module(x)
    # Inspect weight type to confirm subclass still there
    w = module.weight
    inner = w.data if isinstance(w, nn.Parameter) else w
    info = f"weight type={type(w).__name__}, data type={type(inner).__name__}"
    print(f"  {label:<26} -> y.shape={tuple(y.shape)} dtype={y.dtype}  |  {info}")
    return y


def ref_dense(module_dense, x):
    return module_dense(x)


# -----------------------------------------------------------------------------
# Check 1: quantize_() walker behavior
# -----------------------------------------------------------------------------
def check1_walker():
    header("Check 1: quantize_() walker targets nn.Linear only")
    ssd = SSDLikeLinear(128, 256).cuda().bfloat16()
    ssd_before = type(ssd.weight).__name__
    quantize_(ssd, Int8WeightOnlyConfig())
    ssd_after = type(ssd.weight).__name__
    print(f"  SSDLikeLinear weight: {ssd_before} -> {ssd_after}")
    if ssd_after == ssd_before:
        print("  [confirmed] quantize_ did NOT convert SSDLikeLinear")
    else:
        print("  [surprising] quantize_ DID convert SSDLikeLinear — revisit plan")

    lin = nn.Linear(128, 256, bias=False).cuda().bfloat16()
    before = type(lin.weight).__name__
    quantize_(lin, Int8WeightOnlyConfig())
    after_param = type(lin.weight).__name__
    after_data = type(lin.weight.data).__name__
    print(f"  nn.Linear weight: Parameter/{before} -> {after_param} (data: {after_data})")
    # torchao typically leaves weight as nn.Parameter but swaps .data for a subclass
    # Or replaces weight with a subclass parameter. Record either way.
    return lin


# -----------------------------------------------------------------------------
# Checks 2+3: storage contract (A)-(D) + SSD-shaped weight
# -----------------------------------------------------------------------------
def check2_storage_contracts(shape_name, in_f, out_f, dense_ref_weight):
    header(f"Check 2/3: storage contracts on SSDLikeLinear {shape_name} ({out_f} x {in_f})")

    x = torch.randn(4, in_f, device="cuda", dtype=torch.bfloat16)

    # --- Baseline dense reference
    dense = SSDLikeLinear(in_f, out_f).cuda().bfloat16()
    with torch.no_grad():
        dense.weight.copy_(dense_ref_weight)
    y_dense = ref_dense(dense, x)

    # Shared: get a quantized weight via dummy nn.Linear route (stable API)
    dummy = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dummy.weight.copy_(dense_ref_weight)
    quantize_(dummy, Int8WeightOnlyConfig())
    qweight_param = dummy.weight  # nn.Parameter wrapping AffineQuantizedTensor (or similar)
    qweight_inner = qweight_param.data

    print(f"  dummy nn.Linear.weight: type(param)={type(qweight_param).__name__}, "
          f"type(param.data)={type(qweight_inner).__name__}")

    # --- (A): set self.weight = qweight_param (the Parameter from dummy) -----
    try:
        mA = SSDLikeLinear(in_f, out_f).cuda().bfloat16()
        # Some torch versions require popping the existing parameter before assigning
        # a new one. Try straight assignment first.
        mA.weight = qweight_param   # reassign Parameter
        yA = check_dispatch(mA, x, "(A) self.weight=Param(AQT)")
        diff_A = (yA.float() - y_dense.float()).abs().max().item()
        cos_A = F.cosine_similarity(yA.float().flatten(),
                                    y_dense.float().flatten(), dim=0).item()
        print(f"      max abs diff vs dense: {diff_A:.4g}  cosine: {cos_A:.6f}")
        A_ok = True
    except Exception as e:
        print(f"  (A) FAILED: {type(e).__name__}: {e}")
        A_ok = False

    # --- (B): register_buffer, no Parameter at all ---------------------------
    try:
        mB = SSDLikeLinear(in_f, out_f).cuda().bfloat16()
        # Remove the parameter
        del mB._parameters["weight"]
        # Assign as plain attribute (not buffer) — F.linear just reads self.weight
        mB.weight = qweight_inner
        yB = check_dispatch(mB, x, "(B) self.weight=AQT (no Param)")
        diff_B = (yB.float() - y_dense.float()).abs().max().item()
        cos_B = F.cosine_similarity(yB.float().flatten(),
                                    y_dense.float().flatten(), dim=0).item()
        print(f"      max abs diff vs dense: {diff_B:.4g}  cosine: {cos_B:.6f}")
        B_ok = True
    except Exception as e:
        print(f"  (B) FAILED: {type(e).__name__}: {e}")
        B_ok = False

    # --- (C): separate attribute + forward branch --------------------------
    # Just confirm that a plain attribute holding AQT runs via F.linear.
    # This is effectively the same as (B); mark as trivially ok if B ok.
    C_ok = B_ok
    print(f"  (C) branch approach: functionally same as (B), skipped separate test.")

    # --- (D): wrapper Parameter via __init_subclass__ not needed; skip.
    print(f"  (D) not tested yet (would need a Parameter subclass).")

    return A_ok, B_ok


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    lin_q = check1_walker()

    # Small toy
    w_toy = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    A_ok, B_ok = check2_storage_contracts("toy", 128, 256, w_toy)

    # Llama-2-7B hidden=4096. TP=2 local shard sizes:
    # - QKV packed local = (32 + 32 + 32)//2 * 128 = 6144 output, in=4096
    # - gate_up packed local = 2 * 11008 // 2 = 11008 output, in=4096
    # - o_proj local (RowParallel) = 4096 out, 4096//2=2048 in
    # - down_proj local (RowParallel) = 4096 out, 11008//2=5504 in

    hidden = 4096
    shapes = [
        ("qkv_packed_tp2",  hidden,            (32+32+32)//2 * 128),
        ("gateup_packed_tp2", hidden,          2 * 11008 // 2),
        ("o_proj_row_tp2",   hidden // 2,      hidden),
        ("down_proj_row_tp2", 11008 // 2,      hidden),
    ]
    results = {}
    for name, in_f, out_f in shapes:
        w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
        A, B = check2_storage_contracts(name, in_f, out_f, w)
        results[name] = (A, B)

    header("Summary")
    print(f"  (A) Param(AQT) assignment:  {'ok' if A_ok else 'FAIL'}  (toy)")
    for name, (A, B) in results.items():
        print(f"    {name:<24}  A={'ok' if A else 'FAIL'}  B={'ok' if B else 'FAIL'}")


if __name__ == "__main__":
    main()
