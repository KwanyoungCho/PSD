"""Real paged-attention parity: independent arenas versus fused forwards."""
import unittest
from types import SimpleNamespace
import torch
from ssd.layers.attention import Attention
from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
from ssd.engine.helpers.batched_tree_executor import BatchedTreeExecutor


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBatchedTreeExecutor(unittest.TestCase):
    @torch.inference_mode()
    def test_separate_prefixes_masks_and_captured_replay(self):
        device = torch.device('cuda:0')
        torch.manual_seed(37)
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(41, 128, device=device, dtype=torch.float16)
                self.position = torch.nn.Embedding(128, 128, device=device, dtype=torch.float16)
                self.query = torch.nn.Linear(128, 896, bias=False, device=device, dtype=torch.float16)
                self.head = torch.nn.Linear(896, 41, bias=False, device=device, dtype=torch.float16)
                self.attn = Attention(14, 64, 64**-.5, 2, draft=True,
                                      speculate=True, draft_async=True)
                self.attn.k_cache = torch.randn(32, 16, 2, 64, device=device, dtype=torch.float16) * .1
                self.attn.v_cache = torch.randn_like(self.attn.k_cache) * .1
                self.calls = 0
            def forward(self, ids, pos):
                self.calls += 1
                x = (self.embedding(ids) + self.position(pos)) * .2
                return self.attn(self.query(x), x, x)
            def logits(self, x, _):
                return self.head(x)
        model = ToyModel()
        cfg = SimpleNamespace(duet_proxy_total_budget=3, duet_p2_seed_count=3,
                              duet_phase1_k=3, duet_phase2_k=3,
                              duet_tree_c_tensor=3, duet_p2_tree_max_nodes=6,
                              duet_tree_policy='dynamic', greedy_only=False)
        arenas = [P2TreeExecutor(model, model.logits, cfg, device, 16, 4,
                                41, 14, 2, 64, materialize_backbone_logits=False)
                  for _ in range(8)]
        for b, ex in enumerate(arenas):
            ex.prime_capture_inputs(1)
            ex.in_root_tok.copy_(torch.tensor([1, 7, 13], device=device) + b)
            ex.in_root_piv.copy_(torch.tensor([.7, .2, .1], device=device))
            ex.in_prefix_len.fill_(5 + b * 2)
            ex.in_rope_base.fill_(5 + b * 2)
            ex.in_temps.fill_(.7)
            for f in range(ex.F):
                ex.in_slot[f].copy_(b * 64 + 5 + b * 2 + ex.round_offsets[f] + ex.lane_w)
                ex.wrappers[1][f]._paged_kv_indices_buf.copy_(
                    torch.arange(1 + ex.canvas_extra_pages, device=device, dtype=torch.int32) + b * 4)
            ex.parity_noise = [torch.empty(ex.W, ex.V, device=device).exponential_()
                               for _ in range(ex.F)]
        snapshots = []
        model.calls = 0
        for ex in arenas:
            ex.run_once(1)
            snapshots.append({name:getattr(ex,name).clone() for name in
                              ['cell_logits', 'out_valid', 'view_tok', 'view_par', 'out_pq_ref']})
        self.assertEqual(model.calls, 24)
        batch = BatchedTreeExecutor(arenas)
        model.calls = 0
        batch.run_once(1)
        self.assertEqual(model.calls, 3)
        def check():
            for ex, ref in zip(arenas, snapshots):
                for name, expected in ref.items():
                    actual = getattr(ex,name)
                    if actual.is_floating_point():
                        self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3), name)
                    else:
                        self.assertTrue(torch.equal(actual, expected), name)
        check()
        batch.capture(1)
        batch.replay(1)
        check()
        # Change one prefix in place. A fixed graph must use the updated
        # pages while leaving the other request's logits untouched.
        model.attn.v_cache[4, :7].add_(.8)
        for b, ex in enumerate(arenas):
            ex.run_once(1)
            for name in snapshots[b]: snapshots[b][name] = getattr(ex,name).clone()
        batch.replay(1)
        check()


if __name__ == '__main__':
    unittest.main()
