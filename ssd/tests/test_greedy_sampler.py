import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd import SamplingParams
from ssd.layers.sampler import Sampler
from ssd.engine.llm_engine import LLMEngine


class TestGreedySampler(unittest.TestCase):
    def test_request_contract(self):
        admitted = []
        engine = SimpleNamespace(config=SimpleNamespace(greedy_only=True),
                                 scheduler=SimpleNamespace(add=admitted.append))
        for params in [SamplingParams(temperature=.7),
                       SamplingParams(temperature=0, draft_temperature=.7)]:
            with self.assertRaisesRegex(ValueError, "greedy_only"):
                LLMEngine.add_request(engine, [1, 2], params)
        LLMEngine.add_request(engine, [1, 2], SamplingParams(temperature=0))
        self.assertEqual(len(admitted), 1)

    def test_argmax_parity_without_rng_consumption(self):
        sampler = Sampler(greedy_only=True)
        torch.manual_seed(5)
        logits = torch.randn(8, 32768)
        temps = torch.zeros(8)
        state = torch.get_rng_state().clone()
        result = sampler(logits, temps)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(result, Sampler()(logits, temps)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_capture_replay_and_native_dtype_argmax(self):
        for dtype in [torch.float16, torch.bfloat16]:
            sampler = Sampler(greedy_only=True)
            logits = torch.randn(8, 128256, dtype=dtype, device="cuda")
            temps = torch.zeros(8, device="cuda")
            sampler(logits, temps)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = sampler(logits, temps)
            for _ in range(3):
                logits.normal_()
                graph.replay()
                self.assertTrue(torch.equal(out, Sampler()(logits, temps)))


if __name__ == "__main__":
    unittest.main()
