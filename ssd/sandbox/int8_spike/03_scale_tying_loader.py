"""Phase 0 sandbox — checks 5, 6, 7.

5. Scale granularity inspection: which axis? compare global vs per-rank scales.
6. Weight tying defense: tie_word_embeddings semantics + untie rule.
7. weight_loader order: float load first, then quantize hook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig


def header(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def inspect_aqt(aqt, label):
    """Print internal state of an AffineQuantizedTensor."""
    print(f"  [{label}]")
    print(f"    type: {type(aqt).__name__}")
    print(f"    shape: {tuple(aqt.shape)}  dtype: {aqt.dtype}")
    # Poke at common attrs
    for attr in ("tensor_impl", "block_size", "shape", "quant_min", "quant_max",
                 "zero_point_domain", "scale", "_layout"):
        if hasattr(aqt, attr):
            v = getattr(aqt, attr)
            try:
                if isinstance(v, torch.Tensor):
                    print(f"    .{attr}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"    .{attr}: {v!r}")
            except Exception:
                print(f"    .{attr}: <repr failed>")
    # The underlying int8 tensor & scale often live under tensor_impl
    if hasattr(aqt, "tensor_impl"):
        ti = aqt.tensor_impl
        print(f"    tensor_impl type: {type(ti).__name__}")
        for attr in ("int_data", "scale", "zero_point", "_layout"):
            if hasattr(ti, attr):
                v = getattr(ti, attr)
                if isinstance(v, torch.Tensor):
                    print(f"      .{attr}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"      .{attr}: {v!r}")


# -----------------------------------------------------------------------------
# Check 5: scale axis + global vs per-rank
# -----------------------------------------------------------------------------
def check5_scale():
    header("Check 5: scale granularity")
    torch.manual_seed(0)
    out_f, in_f = 512, 256   # W = [out, in]
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.1

    # --- Global quantize
    dummy_g = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dummy_g.weight.copy_(W)
    quantize_(dummy_g, Int8WeightOnlyConfig())
    aqt_g = dummy_g.weight.data
    inspect_aqt(aqt_g, "global (full W)")

    # --- Simulated ColumnParallel shard (split dim=0, keep full dim=1)
    tp = 2
    shard_c = W[: out_f // tp, :].contiguous()
    dummy_c = nn.Linear(in_f, out_f // tp, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dummy_c.weight.copy_(shard_c)
    quantize_(dummy_c, Int8WeightOnlyConfig())
    inspect_aqt(dummy_c.weight.data, f"ColumnShard rank0 [{out_f//tp}, {in_f}]")

    # --- Simulated RowParallel shard (full dim=0, split dim=1)
    shard_r = W[:, : in_f // tp].contiguous()
    dummy_r = nn.Linear(in_f // tp, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dummy_r.weight.copy_(shard_r)
    quantize_(dummy_r, Int8WeightOnlyConfig())
    inspect_aqt(dummy_r.weight.data, f"RowShard rank0 [{out_f}, {in_f//tp}]")

    # Extract scale tensors: torchao stores them under tensor_impl.scale
    def get_scale(aqt):
        return aqt.tensor_impl.scale

    s_g = get_scale(aqt_g)
    s_c = get_scale(dummy_c.weight.data)
    s_r = get_scale(dummy_r.weight.data)
    print("\n  scale shapes:")
    print(f"    global    : {tuple(s_g.shape)}")
    print(f"    column_sh : {tuple(s_c.shape)}")
    print(f"    row_sh    : {tuple(s_r.shape)}")

    # For ColumnParallel: local scales should equal corresponding prefix of global
    # Actually local scale is derived from local shard which IS a prefix of rows.
    # If scale is per-output-channel, local scale for the first out_f/tp channels
    # should match global scale of those channels.
    print(f"\n  column vs global (first {out_f//tp} rows):")
    if s_c.numel() == out_f // tp:
        diff_c = (s_c.float().flatten() - s_g.float().flatten()[: out_f // tp]).abs().max().item()
        print(f"    max abs diff: {diff_c:.6g}  (should be 0)")

    # For RowParallel: local scale is from 1/tp of each row. Expected:
    #   local scale ≤ global scale per output channel
    print(f"\n  row vs global (output-channel scale comparison):")
    if s_r.numel() == out_f:
        g_flat = s_g.float().flatten()
        r_flat = s_r.float().flatten()
        le_count = (r_flat <= g_flat + 1e-6).sum().item()
        print(f"    local scale ≤ global scale count: {le_count}/{out_f}")
        ratio = (r_flat / g_flat).mean().item()
        print(f"    mean(local/global) = {ratio:.4f} (expect < 1)")


# -----------------------------------------------------------------------------
# Check 6: weight tying defense
# -----------------------------------------------------------------------------
def check6_tying():
    header("Check 6: weight tying defense")
    V, D = 1024, 128

    # Simulate: embed shares data with lm_head (as llama3.py:333-334 does)
    embed = nn.Embedding(V, D).cuda().bfloat16()
    lm_head = nn.Linear(D, V, bias=False).cuda().bfloat16()
    # tie
    lm_head.weight.data = embed.weight.data
    print(f"  before: id(embed.data)={id(embed.weight.data)} "
          f"id(lm_head.data)={id(lm_head.weight.data)}  same={embed.weight.data.data_ptr() == lm_head.weight.data.data_ptr()}")

    # A) If we quantize lm_head without untying, does F.embedding still work?
    lm_head_naive = nn.Linear(D, V, bias=False).cuda().bfloat16()
    embed_naive = nn.Embedding(V, D).cuda().bfloat16()
    lm_head_naive.weight.data = embed_naive.weight.data
    try:
        quantize_(lm_head_naive, Int8WeightOnlyConfig())
        # Now test F.embedding on the tied embed_naive
        idx = torch.tensor([0, 1, 2], device="cuda")
        try:
            out = F.embedding(idx, embed_naive.weight)
            print(f"  (naive quantize, no untie): F.embedding STILL works, "
                  f"out shape={tuple(out.shape)}, dtype={out.dtype}")
            # If it works: torchao didn't alias the underlying data?
            print(f"    id(embed.data)={id(embed_naive.weight.data)}  type={type(embed_naive.weight.data).__name__}")
        except Exception as e:
            print(f"  (naive quantize, no untie): F.embedding FAILED "
                  f"{type(e).__name__}: {e}")
    except Exception as e:
        print(f"  naive quantize itself FAILED: {type(e).__name__}: {e}")

    # B) Proper defense: untie before quantize
    lm_head_def = nn.Linear(D, V, bias=False).cuda().bfloat16()
    embed_def = nn.Embedding(V, D).cuda().bfloat16()
    lm_head_def.weight.data = embed_def.weight.data   # tied initially
    # untie: reassign with a clone
    lm_head_def.weight = nn.Parameter(lm_head_def.weight.data.clone())
    print(f"  after untie: same_data_ptr={embed_def.weight.data.data_ptr() == lm_head_def.weight.data.data_ptr()}")
    quantize_(lm_head_def, Int8WeightOnlyConfig())
    # embed_def.weight should still be float bf16
    idx = torch.tensor([0, 1, 2], device="cuda")
    try:
        out = F.embedding(idx, embed_def.weight)
        print(f"  (with untie defense) F.embedding ok, out shape={tuple(out.shape)}, dtype={out.dtype}")
    except Exception as e:
        print(f"  (with untie defense) F.embedding FAILED: {type(e).__name__}: {e}")


# -----------------------------------------------------------------------------
# Check 7: weight_loader-style float load first, then quantize
# -----------------------------------------------------------------------------
def check7_loader_order():
    header("Check 7: weight_loader ordering — float load then quantize")
    torch.manual_seed(0)
    in_f, out_f = 128, 256

    # Mimic ColumnParallelLinear: self.weight = nn.Parameter(empty(out/tp, in))
    class MockCP(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_f, in_f,
                                                   device="cuda", dtype=torch.bfloat16))
        def forward(self, x):
            return F.linear(x, self.weight)

        def weight_loader(self, param, loaded):
            # copy of SSD's pattern: param.data.copy_(loaded_weight.narrow(...))
            param.data.copy_(loaded)

    mod = MockCP()
    full = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
    # Step 1: float load
    mod.weight_loader(mod.weight, full)
    assert torch.equal(mod.weight.data, full), "float load failed"
    print("  step 1 (float load via param.data.copy_): ok")

    # Step 2: quantize hook replaces weight
    dummy = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dummy.weight.copy_(mod.weight.data)
    quantize_(dummy, Int8WeightOnlyConfig())
    mod.weight = dummy.weight
    print(f"  step 2 (replace with AQT Param): ok. "
          f"type={type(mod.weight.data).__name__}")

    # Step 3: forward should still work
    x = torch.randn(4, in_f, device="cuda", dtype=torch.bfloat16)
    y = mod(x)
    print(f"  step 3 (forward after replacement): ok, y shape={tuple(y.shape)}")

    # Step 4: what if loader is called AGAIN after quantization? This should not
    # happen in practice (load happens once), but confirm it fails loudly.
    try:
        mod.weight_loader(mod.weight, full)   # param.data.copy_ on AQT
        print("  step 4 (re-load after quant): UNEXPECTEDLY succeeded — "
              "subclass .data.copy_ is silently permissive")
    except Exception as e:
        print(f"  step 4 (re-load after quant): FAILED as expected — "
              f"{type(e).__name__}: {e}")


def main():
    check5_scale()
    check6_tying()
    check7_loader_order()


if __name__ == "__main__":
    main()
