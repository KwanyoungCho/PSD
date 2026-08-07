"""CPU contracts for DUET Phase-1 dynamic tree preparation."""
import unittest

import torch

from ssd.engine.helpers.p1_tree import (
    build_uniform_p1_roots, choose_p1_context_bucket,
    p1_context_buckets)
from ssd.utils.async_helpers.async_spec_helpers import (
    compute_megaspec_lookahead)


class TestP1UniformRoots(unittest.TestCase):
    def test_uniform_topk_scores_padding_and_glue_rows(self):
        logits = torch.tensor([
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 5.0, 4.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ])
        # row0 excludes returned[1]=5; row1 excludes returned[2]=4.
        out = build_uniform_p1_roots(
            logits, torch.tensor([1, 5, 4]), 2, torch.tensor([1.0]),
            sampler_x=None, async_fan_out=3, root_width=8)
        self.assertEqual(out["real_roots"], 6)
        self.assertEqual(out["width"], 8)
        self.assertEqual(out["tokens"].tolist(), [4, 3, 5, 3, 0, 1, 0, 0])
        self.assertEqual(
            out["context_ids"].tolist(), [0, 0, 1, 1, 2, 2, 0, 0])
        self.assertEqual(
            out["valid"].tolist(), [True] * 6 + [False, False])
        self.assertEqual(out["glue_rows"].tolist(), [
            [1, 0, 0], [1, 0, 0],
            [1, 1, 0], [1, 1, 0],
            [1, 1, 1], [1, 1, 1],
            [0, 0, 0], [0, 0, 0],
        ])
        probs = torch.softmax(logits, dim=-1)
        expected = torch.tensor([
            probs[0, 4], probs[0, 3], probs[1, 5], probs[1, 3],
            probs[2, 0], probs[2, 1], 0.0, 0.0])
        self.assertTrue(torch.allclose(out["scores"], expected))

    def test_custom_tree_context_visibility_is_repeated_per_root(self):
        logits = torch.arange(15, dtype=torch.float32).view(3, 5)
        rows = torch.tensor([[1, 0, 0, 0],
                             [1, 1, 0, 1],
                             [1, 0, 1, 0]], dtype=torch.uint8)
        out = build_uniform_p1_roots(
            logits, torch.tensor([0, 1, 2]), 2, torch.ones(3),
            sampler_x=None, async_fan_out=3,
            context_glue_rows=rows)
        self.assertTrue(torch.equal(
            out["glue_rows"], rows.repeat_interleave(2, dim=0)))

    def test_root_width_must_hold_every_uniform_candidate(self):
        with self.assertRaises(ValueError):
            build_uniform_p1_roots(
                torch.zeros(3, 8), torch.zeros(3, dtype=torch.int64),
                2, torch.ones(3), sampler_x=None, async_fan_out=3,
                root_width=5)


class TestP1ShapeBuckets(unittest.TestCase):
    def test_buckets_cover_chain_and_tree_contexts(self):
        buckets = p1_context_buckets(9, 4, 13, 8)
        self.assertIn(10, buckets)  # K1 chain contexts
        self.assertIn(14, buckets)  # max P1 tree + recovery
        self.assertEqual(buckets, (10, 14))
        self.assertEqual(choose_p1_context_bucket(5, buckets), 10)
        self.assertEqual(choose_p1_context_bucket(8, buckets), 10)
        self.assertEqual(choose_p1_context_bucket(10, buckets), 10)
        with self.assertRaises(ValueError):
            choose_p1_context_bucket(15, buckets)

    def test_scheduler_reserves_common_tree_glue_and_p1_canvas(self):
        # Champion-like P1: 14 contexts * 2 roots, 9 rounds.  The dynamic
        # P1 canvas dominates P2 and the 14-row tree glue exceeds K1+1.
        self.assertEqual(compute_megaspec_lookahead(
            0, 13, split_k1k2=True, K1=9, K2=4,
            mq_p1=28, mq_p2=10, glue_width=14), 14 + 9 * 28)

    def test_two_x_depth_budget_uses_nineteen_context_canvas(self):
        buckets = p1_context_buckets(9, 4, 18, 8)
        self.assertEqual(buckets, (10, 19))
        self.assertEqual(choose_p1_context_bucket(19, buckets), 19)
        self.assertEqual(compute_megaspec_lookahead(
            0, 13, split_k1k2=True, K1=9, K2=4,
            mq_p1=38, mq_p2=10, glue_width=19), 19 + 9 * 38)


class TestAsyncResponseEnvelope(unittest.TestCase):
    def test_token_envelope_widens_without_widening_logits(self):
        from types import SimpleNamespace
        from ssd.engine.helpers.p2_tree import tree_wire_ints_len
        from ssd.engine.speculator_async import SpeculatorAsync

        spec = SpeculatorAsync.__new__(SpeculatorAsync)
        spec.device = torch.device("cpu")
        spec.K = 13
        spec.response_width = 18
        spec.async_fan_out = 3
        spec.max_blocks = 8
        spec.vocab_size = 32
        spec.draft_dtype = torch.float16
        spec.config = SimpleNamespace(
            duet_tree_enabled=True,
            duet_tree_wire_nodes=18,
        )
        spec._alloc_handshake_bufs(2)

        tree_extra = 2 * tree_wire_ints_len(18)
        self.assertEqual(
            spec._fused_response.numel(), 3 * 2 + 2 * 18 + tree_extra)
        self.assertEqual(tuple(spec._logits_q.shape), (2, 13, 32))
        self.assertEqual(tuple(spec._tree_parent_q.shape), (2, 18, 32))


if __name__ == "__main__":
    unittest.main()
