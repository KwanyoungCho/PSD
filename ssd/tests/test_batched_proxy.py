import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd.engine.verifier import Verifier
from ssd.engine.helpers.batched_proxy import (
    batched_chain_candidates, BatchedChainProxyCUDAGraph)


def eager_wire(e, q, tokens, vk, pack=False):
    cfg = SimpleNamespace(
        duet_proxy_top_k=8, duet_proxy_wire_N=18,
        duet_proxy_on_draft=False, duet_exit_replica=False,
        duet_policy="b", jit_speculate=True,
        duet_tree_enabled=pack, duet_tree_policy="dynamic" if pack else "off",
        duet_p2_budget_at=lambda k: k+1)
    stub = SimpleNamespace(target_model_runner=SimpleNamespace(config=cfg))
    result = []
    with patch.object(Verifier, "_send_proxy_wire",
                      side_effect=lambda *a: result.extend([a[-2].clone(), a[-1].clone()])):
        Verifier._compute_and_send_proxy(stub, e, tokens, q, len(e), q.shape[1],
                                        None, 0, valid_k=vk)
    return [x.view(len(e), 18) for x in result]


class TestBatchedProxy(unittest.TestCase):
    def test_cpu_ragged_matches_production_eager_policy(self):
        for b in (2, 3, 4, 8):
            for pack in (False, True):
                torch.manual_seed(b)
                e, q = torch.randn(b, 5, 257), torch.randn(b, 4, 257)
                y = torch.randint(257, (b, 4))
                vk = torch.tensor([4 if i % 2 else 2 for i in range(b)])
                ref = eager_wire(e, q, y, vk, pack)
                out = batched_chain_candidates(e, q, y, vk, 8, 18, pack)
                self.assertTrue(torch.equal(ref[0], out[0]))
                self.assertTrue(torch.equal(ref[1], out[1]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graph_replay_ragged_padding_and_changing_batch(self):
        graph = BatchedChainProxyCUDAGraph(4, 4, 128256, 8, 18, True,
                                          torch.bfloat16, "cuda")
        for b in (4, 3, 2, 4):
            torch.manual_seed(b)
            e = torch.randn(b, 5, 128256, dtype=torch.bfloat16, device="cuda")
            q = torch.randn(b, 4, 128256, dtype=torch.bfloat16, device="cuda")
            y = torch.randint(128256, (b, 4), device="cuda")
            vk = torch.tensor([4 if i % 2 else 2 for i in range(b)], device="cuda")
            ref = eager_wire(e, q, y, vk, True)
            out = graph.replay(e, q, y, vk)
            self.assertTrue(torch.equal(ref[0], out[0]))
            self.assertTrue(torch.equal(ref[1], out[1]))


if __name__ == "__main__":
    unittest.main()
