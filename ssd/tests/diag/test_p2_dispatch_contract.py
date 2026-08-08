"""Fast CPU tests for the SGLang-style P2 dispatch/input contract."""
import types
import unittest

import torch

from ssd.engine.draft_runner import (
    DraftRunner, _P2StepState, _P2_EXECUTOR_POLICIES,
)
from ssd.engine.helpers.p2_tree import sanitize_root_inputs


class TestP2RootContract(unittest.TestCase):
    def test_public_dynamic_policy_is_executor_supported(self):
        self.assertIn("dynamic", _P2_EXECUTOR_POLICIES)

    def test_invalid_roots_become_safe_inactive_lanes(self):
        toks = torch.tensor([7, -1, 99, 8])
        piv = torch.tensor([0.7, 0.2, 0.1, float("nan")])
        rope = torch.tensor([10, 11, 12, 13])
        st, sp, sr, valid = sanitize_root_inputs(
            toks, piv, rope, vocab_size=32, max_position=128)
        self.assertTrue(torch.equal(valid, torch.tensor(
            [True, False, False, False])))
        self.assertTrue(torch.equal(st, torch.tensor([7, 0, 0, 0])))
        self.assertTrue(torch.equal(sr, torch.tensor([10, 0, 0, 0])))
        self.assertTrue(torch.equal(sp, torch.tensor([0.7, 0.0, 0.0, 0.0])))


class TestP2StepState(unittest.TestCase):
    def _runner(self):
        r = object.__new__(DraftRunner)
        r.device = torch.device("cpu")
        r.block_size = 16
        r.config = types.SimpleNamespace(max_model_len=128)
        r.kv_cache = torch.zeros(2, 1, 8, 16, 1, 1)
        return r

    def _inputs(self):
        return dict(
            rope0=torch.tensor([30, 31]),
            step_slot_maps=[torch.tensor([30, 31]),
                            torch.tensor([32, 33])],
            step_context_lens=[torch.tensor([[32]]),
                               torch.tensor([[34]])],
            dbt=torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]],
                             dtype=torch.int32),
            F=2,
            W=2,
        )

    def test_active(self):
        state = self._runner()._p2_step_state(**self._inputs())
        self.assertTrue(state.active)
        self.assertEqual(state.ctx_len, 32)

    def test_negative_active_slot_is_idle(self):
        args = self._inputs()
        args["step_slot_maps"][1][0] = -1
        state = self._runner()._p2_step_state(**args)
        self.assertFalse(state.active)
        self.assertEqual(state.reason, "slot")

    def test_padded_root_sentinel_is_not_a_real_root(self):
        args = self._inputs()
        args["rope0"] = torch.tensor([30, -1])
        args["root_count"] = 1
        state = self._runner()._p2_step_state(**args)
        self.assertTrue(state.active)

    def test_empty_root_set_is_idle(self):
        args = self._inputs()
        args["root_count"] = 0
        state = self._runner()._p2_step_state(**args)
        self.assertFalse(state.active)
        self.assertEqual(state.reason, "no_roots")


class TestP2SingleDispatch(unittest.TestCase):
    def test_active_step_calls_rollout_once(self):
        r = object.__new__(DraftRunner)
        r.device = torch.device("cpu")
        r.config = types.SimpleNamespace(duet_phase2_k=1, duet_tree_nv=2)
        r.hf_config = types.SimpleNamespace(
            vocab_size=8, torch_dtype=torch.float32)
        r._compute_step_positions_and_slot_maps = lambda *a, **k: (
            None, None, [torch.tensor([[4]])], [torch.tensor([1, 2])])
        r._p2_step_state = lambda *a, **k: _P2StepState(ctx_len=4)
        calls = {"n": 0}
        views = {"valid": torch.tensor([1])}
        tokens = torch.tensor([[3]])
        logits = torch.zeros(1, 1, 8)

        def rollout(*args, **kwargs):
            calls["n"] += 1
            return views, (tokens, logits), logits

        r._p2tree_rollout = rollout
        layout = types.SimpleNamespace(MQ_LEN=1)
        tree_args = {
            "metadata_ints": (1, 1, 1, 1),
            "positions": torch.tensor([1]),
            "rope_positions": torch.tensor([1]),
            "block_tables": torch.tensor([[0]], dtype=torch.int32),
        }
        got_t, got_l = r._run_p2_tree_step(
            {}, torch.tensor([[1]]), torch.tensor([[1]]),
            torch.tensor([[0.5]]), tree_args, layout,
            torch.tensor([0.8]))
        self.assertEqual(calls["n"], 1)
        self.assertIs(r._tree_views, views)
        self.assertIs(got_t, tokens)
        self.assertIs(got_l, logits)


class TestP2ExecutorOutputContract(unittest.TestCase):
    def _runner(self):
        r = object.__new__(DraftRunner)
        r.config = types.SimpleNamespace(duet_proxy_total_budget=10)
        return r

    def _executor(self, rows):
        return types.SimpleNamespace(
            W=10,
            out_valid=torch.zeros(rows, dtype=torch.int64),
            view_tok=torch.zeros(rows, 8, dtype=torch.int64),
            view_par=torch.full((rows, 8), -1, dtype=torch.int64),
            view_sib=torch.zeros(rows, 8, dtype=torch.int64),
            view_rawq=torch.zeros(rows, 8),
            view_pcell=torch.full((rows, 8), -1, dtype=torch.int64),
            out_pq_ref=torch.full((rows, 8), -1, dtype=torch.int64),
            out_pq_cells=torch.full((rows, 8), -1, dtype=torch.int64),
            out_u_valid=torch.zeros(rows, dtype=torch.int64),
            cell_logits=torch.zeros(40, 32),
            out_backbone_tok=torch.zeros(rows, 4, dtype=torch.int64),
            out_backbone_logits=torch.zeros(rows, 4, 32),
        )

    def test_rejects_root_width_payload_beside_layout_width_keys(self):
        with self.assertRaisesRegex(RuntimeError, "output/layout row"):
            self._runner()._exec_outputs_to_views(self._executor(6), 6)

    def test_accepts_layout_width_with_invalid_padding_rows(self):
        views, (tokens, logits), _ = self._runner() \
            ._exec_outputs_to_views(self._executor(10), 6)
        self.assertEqual(tuple(views["valid"].shape), (10,))
        self.assertEqual(tuple(tokens.shape), (10, 4))
        self.assertEqual(tuple(logits.shape), (10, 4, 32))


if __name__ == "__main__":
    unittest.main()
