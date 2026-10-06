import os
import unittest
os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd.layers.layernorm import RMSDNorm, prepare_norms_for_colocation


class TestColocatedNorm(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_two_eps_values_are_capturable_and_correct(self):
        prepare_norms_for_colocation()
        # Reproduce the engine's default CUDA device and dtype context.
        old = torch.get_default_device()
        torch.set_default_device("cuda")
        try:
            for dim, eps in [(4096, 1e-5), (1024, 1e-6)]:
                norm = RMSDNorm(dim, eps).to(dtype=torch.bfloat16, device="cuda")
                x = torch.randn(8, dim, dtype=torch.bfloat16, device="cuda")
                with torch.inference_mode():
                    for _ in range(3):
                        norm(x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = norm(x)
                    graph.replay()
                    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)+eps)).to(x.dtype)*norm.weight
                    torch.testing.assert_close(out, expected, rtol=.01, atol=.01)
        finally:
            torch.set_default_device(old)


if __name__ == "__main__":
    unittest.main()
