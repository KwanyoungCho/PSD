"""CPU contracts for DUET Phase-1 dynamic tree preparation."""
import unittest

import torch

from ssd.engine.draft_runner import _should_run_p2_tree

from ssd.engine.helpers.p1_tree import (
    build_uniform_p1_roots, choose_p1_context_bucket,
    p1_context_buckets)
from ssd.engine.helpers.p2_tree import (
    q_probs_from_logits, selected_q_probs_from_logits)
from ssd.utils.async_helpers.async_spec_helpers import (
    compute_megaspec_lookahead, compute_tree_forward_width)


class TestP1UniformRoots(unittest.TestCase):
    def test_selected_q_matches_full_distribution(self):
        g = torch.Generator().manual_seed(7)
        logits = torch.randn(4, 31, generator=g)
        temps = torch.tensor([0.4, 0.7, 1.0, 1.3])
        ids = torch.randint(0, 31, (4, 5), generator=g)
        for sampler_x in (None, 0.6, 1.4):
            full = q_probs_from_logits(
                logits.clone(), temps, sampler_x=sampler_x, F=3)
            selected = selected_q_probs_from_logits(
                logits, temps, ids, sampler_x=sampler_x, F=3)
            self.assertTrue(torch.allclose(
                selected, full.gather(1, ids), atol=2e-7, rtol=2e-6))
            rows = torch.tensor([
                [0, 1, 2, 3, 0], [3, 2, 1, 0, 3],
                [1, 1, 2, 2, 3], [2, 0, 3, 1, 0]])
            paired = selected_q_probs_from_logits(
                logits, temps, ids, sampler_x=sampler_x, F=3,
                source_rows=rows)
            self.assertTrue(torch.allclose(
                paired, full[rows, ids], atol=2e-7, rtol=2e-6))

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
        reach = torch.tensor([
            1.0, probs[0, 5], probs[0, 5] * probs[1, 4]])
        expected = torch.tensor([
            probs[0, 4], probs[0, 3], probs[1, 5], probs[1, 3],
            probs[2, 0], probs[2, 1], 0.0, 0.0])
        expected[:6] *= reach.repeat_interleave(2)
        self.assertTrue(torch.allclose(out["scores"], expected))
        self.assertTrue(torch.allclose(out["context_reach"], reach))

    def test_context_reach_discounts_late_roots(self):
        # The next-token path has probability 0.1 at each edge.  A late root
        # with the same local confidence must not outrank an early root just
        # because the old implementation ignored whether its context is
        # likely to be reached.
        logits = torch.log(torch.tensor([
            [.1, .1, .4, .4],
            [.1, .1, .4, .4],
            [.1, .1, .4, .4],
        ]))
        out = build_uniform_p1_roots(
            logits, torch.tensor([0, 0, 0]), 1, torch.ones(3),
            sampler_x=None, async_fan_out=3)
        self.assertTrue(torch.allclose(
            out["context_reach"], torch.tensor([1.0, .1, .01]),
            atol=1e-6))
        self.assertGreater(float(out["scores"][0]),
                           float(out["scores"][1]))
        self.assertGreater(float(out["scores"][1]),
                           float(out["scores"][2]))

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

    def test_root_selection_excludes_returned_without_mutating_logits(self):
        logits = torch.tensor([
            [9.0, 8.0, 7.0, 6.0, 5.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ])
        before = logits.clone()
        # row0 excludes returned[1]=0, so its next two choices are 1 and 2;
        # the final row excludes nothing and keeps 4 and 3.
        out = build_uniform_p1_roots(
            logits, torch.tensor([4, 0]), 2, torch.ones(2),
            sampler_x=None, async_fan_out=3)
        self.assertEqual(out["tokens"].tolist(), [1, 2, 4, 3])
        self.assertTrue(torch.equal(logits, before))

    def test_persistent_output_buffers_are_reused(self):
        logits = torch.arange(18, dtype=torch.float32).view(3, 6)
        buffers = {
            "tokens": torch.empty(8, dtype=torch.int64),
            "scores": torch.empty(8, dtype=torch.float32),
            "context_ids": torch.empty(8, dtype=torch.int64),
            "valid": torch.empty(8, dtype=torch.bool),
            "glue_rows": torch.empty(8, 3, dtype=torch.uint8),
            "context_reach": torch.empty(3, dtype=torch.float32),
        }
        out = build_uniform_p1_roots(
            logits, torch.tensor([0, 1, 2]), 2, torch.ones(3),
            sampler_x=None, async_fan_out=3, root_width=8,
            output_buffers=buffers)
        for name, buf in buffers.items():
            self.assertEqual(out[name].data_ptr(), buf.data_ptr(), name)
        self.assertEqual(out["valid"].tolist(),
                         [True] * 6 + [False, False])


class TestP1ShapeBuckets(unittest.TestCase):
    def test_both_trees_reserve_draft_graph_headroom(self):
        import dataclasses
        from ssd.engine.draft_runner import DraftRunner

        @dataclasses.dataclass
        class _Cfg:
            model: str = "target"
            draft: str = "draft"
            draft_async: bool = True
            duet_p1_tree_policy: str = "off"
            duet_p2_tree_policy: str = "off"
            gpu_memory_utilization: float = 0.7
            tokenizer_path: str | None = None
            d_model_target: int | None = None
            use_eagle: bool = False
            hf_config: object | None = None
            enforce_eager: bool = False

        chain = DraftRunner.create_draft_config(_Cfg())
        p1_only = DraftRunner.create_draft_config(
            _Cfg(duet_p1_tree_policy="on"))
        both = DraftRunner.create_draft_config(_Cfg(
            duet_p1_tree_policy="on", duet_p2_tree_policy="on"))
        sync = DraftRunner.create_draft_config(_Cfg(draft_async=False))
        self.assertEqual(chain.gpu_memory_utilization, 0.8)
        self.assertEqual(p1_only.gpu_memory_utilization, 0.8)
        self.assertEqual(both.gpu_memory_utilization, 0.75)
        self.assertEqual(sync.gpu_memory_utilization, 0.75)

    def test_phase_switches_select_dynamic_for_p1_only(self):
        from ssd.config import _normalize_tree_switches

        self.assertEqual(
            _normalize_tree_switches("off", "off", None),
            ("off", "off", "off"))
        self.assertEqual(
            _normalize_tree_switches("on", "off", None),
            ("on", "off", "dynamic"))
        self.assertEqual(
            _normalize_tree_switches("off", "on", None),
            ("off", "on", "dynamic"))
        self.assertEqual(
            _normalize_tree_switches("on", "on", None),
            ("on", "on", "dynamic"))
        # Draft-config reconstruction passes the normalized selector back
        # through Config.  It must not silently enable P2.
        self.assertEqual(
            _normalize_tree_switches("on", "off", "dynamic"),
            ("on", "off", "dynamic"))
        self.assertEqual(
            _normalize_tree_switches("off", "off", "eagle"),
            ("off", "on", "eagle"))

    def test_buckets_cover_chain_and_tree_contexts(self):
        buckets = p1_context_buckets(9, 4, 13, 8)
        self.assertIn(5, buckets)   # K2 short-chain contexts
        self.assertIn(10, buckets)  # K1 chain contexts
        self.assertIn(14, buckets)  # max P1 tree + recovery
        self.assertEqual(buckets, (5, 10, 14))
        self.assertEqual(choose_p1_context_bucket(5, buckets), 5)
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
        self.assertEqual(buckets, (5, 10, 19))
        self.assertEqual(choose_p1_context_bucket(19, buckets), 19)
        self.assertEqual(compute_megaspec_lookahead(
            0, 13, split_k1k2=True, K1=9, K2=4,
            mq_p1=38, mq_p2=10, glue_width=19), 19 + 9 * 38)

    def test_variable_width_p1_reserves_compact_cell_sum(self):
        # K1=9, largest P1 context: round widths [38,16x8] = 166.
        # P2 is only 4x10=40, so the exact P1 footprint dominates.
        self.assertEqual(compute_megaspec_lookahead(
            0, 13, split_k1k2=True, K1=9, K2=4,
            mq_p1=16, mq_p2=10, glue_width=19,
            cells_p1=38 + 8 * 16, cells_p2=4 * 10), 19 + 166)

    def test_p1_only_never_dispatches_p2_tree(self):
        class _Cfg:
            # The legacy aggregate is deliberately non-off when only P1 is
            # enabled.  Phase dispatch must ignore it.
            duet_tree_policy = "dynamic"
            duet_p1_tree_policy = "on"
            duet_p2_tree_policy = "off"

        temps = torch.tensor([0.7])
        self.assertFalse(_should_run_p2_tree(_Cfg(), 1, temps))
        _Cfg.duet_p2_tree_policy = "on"
        self.assertTrue(_should_run_p2_tree(_Cfg(), 1, temps))
        self.assertFalse(_should_run_p2_tree(_Cfg(), 2, temps))
        self.assertFalse(_should_run_p2_tree(
            _Cfg(), 1, torch.tensor([0.0])))

    def test_wide_p1_canvas_reservation_crosses_page_safely(self):
        # N1=18 exposes 19 contexts.  Two roots per context gives R=38,
        # while scale=1.25 replays W=48 cells for each of the nine rounds.
        # Reserving the old R-only footprint under-allocates near a 256-token
        # page boundary.
        width = compute_tree_forward_width(19 * 2, 1.25)
        self.assertEqual(width, 48)
        self.assertEqual(compute_megaspec_lookahead(
            0, 13, split_k1k2=True, K1=9, K2=4,
            mq_p1=width, mq_p2=10, glue_width=19), 19 + 9 * 48)

    def test_glue_positions_slice_wide_response_buffer(self):
        """A response buffer wider than K must not widen a K-node glue."""
        from types import SimpleNamespace
        from ssd.engine.draft_runner import DraftRunner

        runner = DraftRunner.__new__(DraftRunner)
        runner.config = SimpleNamespace(speculate_k=13, use_eagle=False)
        runner.device = torch.device("cpu")
        runner.block_size = 256
        runner._arange_kp1 = torch.arange(19, dtype=torch.int64)
        dbt = torch.arange(8, dtype=torch.int32).view(1, 8)
        num_tokens = torch.tensor([512], dtype=torch.int64)

        # Regression case from the real-model P1=18 smoke: n_valid happened
        # to equal speculate_k (13), while the backing buffer had 19 slots.
        for valid_k in (13, 18):
            ctxt = runner.prepare_glue_decode_ctxt(
                num_tokens=num_tokens,
                input_ids=torch.zeros(valid_k + 1, dtype=torch.int64),
                dbt=dbt,
                B=1,
                valid_k=valid_k,
            )
            self.assertEqual(ctxt["positions"].numel(), valid_k + 1)
            self.assertEqual(ctxt["slot_map"].numel(), valid_k + 1)
            self.assertEqual(ctxt["max_seqlen_q"], valid_k + 1)


class TestAsyncResponseEnvelope(unittest.TestCase):
    def test_batched_lookup_cannot_downgrade_tree_rows_to_chain(self):
        from ssd.engine.helpers.p2_tree import \
            filter_unservable_tree_matches

        match = torch.tensor([[True, True, False],
                              [False, True, True]])
        kinds = torch.tensor([False, True, False])
        self.assertTrue(torch.equal(
            filter_unservable_tree_matches(match, kinds, 2),
            torch.tensor([[True, False, False],
                          [False, False, True]])))
        # A B=1 response can carry the topology/parent-q sidecar and keeps
        # the exact original match matrix.
        self.assertIs(filter_unservable_tree_matches(match, kinds, 1), match)
        with self.assertRaises(ValueError):
            filter_unservable_tree_matches(match, kinds[:2], 2)

    def test_logit_payloads_are_mutually_exclusive(self):
        from ssd.engine.helpers.p2_tree import tree_response_logit_rows

        self.assertEqual(
            tree_response_logit_rows(0, 0, 13, 18, 8), (13, 0))
        self.assertEqual(
            tree_response_logit_rows(14, 1, 13, 18, 8), (0, 14))
        self.assertEqual(
            tree_response_logit_rows(7, 2, 13, 18, 8), (0, 7))
        with self.assertRaises(ValueError):
            tree_response_logit_rows(9, 2, 13, 18, 8)
        with self.assertRaises(ValueError):
            tree_response_logit_rows(1, 0, 13, 18, 8)

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
