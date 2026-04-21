"""Phase 0 sandbox — checks 4 + 8.

4. CUDA graph capture/replay with F.linear + AffineQuantizedTensor weight.
8. Coexistence with @torch.compile modules and @torch.inference_mode().

Strategy:
  - Warmup first (force triton autotune if any).
  - Capture in a torch.cuda.graph() context.
  - Replay N times, compare against eager run.
  - Also run inside @torch.inference_mode().
  - Also test side-by-side with a @torch.compile'd module (e.g., a mock RMSNorm).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization import quantize_, Int8WeightOnlyConfig


def header(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


class SSDLikeLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return F.linear(x, self.weight, None)


def make_quantized(in_f, out_f, ref_weight):
    """Produce a SSDLikeLinear whose self.weight is Param(AQT) — contract (A)."""
    mod = SSDLikeLinear(in_f, out_f).cuda().bfloat16()
    dummy = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    with torch.no_grad():
        dummy.weight.copy_(ref_weight)
    quantize_(dummy, Int8WeightOnlyConfig())
    mod.weight = dummy.weight   # Parameter containing AffineQuantizedTensor
    return mod


# ------------------------------------------------------------------
# Check 4: CUDA graph capture/replay
# ------------------------------------------------------------------
def check4_graph(in_f=4096, out_f=11008, batch=4):
    header(f"Check 4: CUDA graph capture/replay ({batch}x{in_f} -> {batch}x{out_f})")
    torch.manual_seed(0)
    ref_w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
    mod = make_quantized(in_f, out_f, ref_w)

    static_x = torch.randn(batch, in_f, device="cuda", dtype=torch.bfloat16)
    static_y = torch.empty(batch, out_f, device="cuda", dtype=torch.bfloat16)

    # ---- Warmup (required before capture: eats triton autotune, allocator setup)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            _ = mod(static_x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # ---- Eager reference for comparison
    y_eager = mod(static_x).clone()

    # ---- Capture
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            static_y = mod(static_x)
        capture_ok = True
        print("  capture: ok")
    except Exception as e:
        capture_ok = False
        print(f"  capture: FAILED {type(e).__name__}: {e}")
        return False

    # ---- Replay N times with different inputs, verify consistency
    max_err = 0.0
    for i in range(10):
        new_x = torch.randn_like(static_x)
        static_x.copy_(new_x)
        g.replay()
        torch.cuda.synchronize()
        y_replay = static_y.clone()
        # Eager run (no graph) for ground truth
        y_ref = mod(new_x)
        err = (y_replay.float() - y_ref.float()).abs().max().item()
        max_err = max(max_err, err)
    print(f"  replay 10x max abs diff replay vs eager: {max_err:.4g}")
    replay_ok = max_err < 1e-2
    if not replay_ok:
        print("  replay: FAIL (diff too large — dispatch may not be graph-safe)")
    else:
        print("  replay: ok")

    return capture_ok and replay_ok


# ------------------------------------------------------------------
# Check 8a: @torch.inference_mode()
# ------------------------------------------------------------------
def check8a_inference_mode():
    header("Check 8a: @torch.inference_mode() compatibility")
    torch.manual_seed(0)
    in_f, out_f = 4096, 11008
    ref_w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
    mod = make_quantized(in_f, out_f, ref_w)
    x = torch.randn(4, in_f, device="cuda", dtype=torch.bfloat16)

    @torch.inference_mode()
    def run(x):
        return mod(x)

    try:
        y1 = run(x)
        y2 = mod(x)   # comparison run without inference_mode
        diff = (y1.float() - y2.float()).abs().max().item()
        print(f"  inference_mode run ok. max diff vs no-inference_mode: {diff:.4g}")
        return diff < 1e-3
    except Exception as e:
        print(f"  inference_mode: FAILED {type(e).__name__}: {e}")
        return False


# ------------------------------------------------------------------
# Check 8b: coexistence with @torch.compile
# ------------------------------------------------------------------
def check8b_torch_compile():
    header("Check 8b: @torch.compile module coexisting with AQT linear")
    torch.manual_seed(0)
    in_f, out_f = 4096, 11008

    # Mock RMSNorm like SSD's layernorm.py pattern: an nn.Module that uses
    # @torch.compile on its forward.
    class CompiledNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d, dtype=torch.bfloat16))
        @torch.compile(dynamic=False, fullgraph=False)
        def forward(self, x):
            var = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + 1e-5)
            return (x * self.weight.float()).to(torch.bfloat16)

    norm = CompiledNorm(in_f).cuda()
    ref_w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
    qlin = make_quantized(in_f, out_f, ref_w)

    x = torch.randn(4, in_f, device="cuda", dtype=torch.bfloat16)

    def run(x):
        h = norm(x)
        return qlin(h)

    try:
        y1 = run(x)   # first call triggers compile
        y2 = run(x)   # second call uses compiled cache
        diff = (y1.float() - y2.float()).abs().max().item()
        print(f"  compile + AQT run ok. max diff call1 vs call2: {diff:.4g}")
        # Should be 0 since input identical
        return diff < 1e-5
    except Exception as e:
        print(f"  compile coexist: FAILED {type(e).__name__}: {e}")
        return False


# ------------------------------------------------------------------
# Check 8c: inference_mode + graph capture combined
# ------------------------------------------------------------------
def check8c_graph_under_inference_mode():
    header("Check 8c: CUDA graph capture under @torch.inference_mode()")
    torch.manual_seed(0)
    in_f, out_f = 4096, 11008
    ref_w = torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16)
    mod = make_quantized(in_f, out_f, ref_w)
    static_x = torch.randn(4, in_f, device="cuda", dtype=torch.bfloat16)
    static_y = torch.empty(4, out_f, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                _ = mod(static_x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                static_y = mod(static_x)
            print("  capture under inference_mode: ok")
        except Exception as e:
            print(f"  capture under inference_mode: FAILED {type(e).__name__}: {e}")
            return False

        # replay
        max_err = 0.0
        for _ in range(5):
            new_x = torch.randn_like(static_x)
            static_x.copy_(new_x)
            g.replay()
            torch.cuda.synchronize()
            y_ref = mod(new_x)
            err = (static_y.float() - y_ref.float()).abs().max().item()
            max_err = max(max_err, err)
        print(f"  replay under inference_mode max diff: {max_err:.4g}")
        return max_err < 1e-2


def main():
    ok4 = check4_graph()
    ok8a = check8a_inference_mode()
    ok8b = check8b_torch_compile()
    ok8c = check8c_graph_under_inference_mode()

    header("Summary")
    print(f"  4   CUDA graph capture/replay:       {'ok' if ok4 else 'FAIL'}")
    print(f"  8a  inference_mode:                  {'ok' if ok8a else 'FAIL'}")
    print(f"  8b  torch.compile coexist:           {'ok' if ok8b else 'FAIL'}")
    print(f"  8c  graph under inference_mode:      {'ok' if ok8c else 'FAIL'}")


if __name__ == "__main__":
    main()
