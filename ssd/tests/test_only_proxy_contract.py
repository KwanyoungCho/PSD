"""CPU regressions for the logical K1=0 (only-proxy) cache ablation."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault("SSD_HF_CACHE", "/home/eslab/models/hub")
os.environ.setdefault("SSD_DATASET_DIR", "/home/eslab/models/processed_datasets")

from ssd.config import Config
from ssd.engine.draft_runner import DraftRunner


def _fake_hf_config(model_path: str):
    return SimpleNamespace(
        max_position_embeddings=(4096 if model_path == "/target" else 2048),
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=80,
        model_type="llama",
        vocab_size=32000,
        rope_theta=10000.0,
    )


class TestOnlyProxyConfig(unittest.TestCase):
    def test_budget_and_wire_exclude_all_phase1_roots(self):
        with patch("ssd.config.os.path.isdir", return_value=True), \
             patch("ssd.config.AutoConfig.from_pretrained",
                   side_effect=_fake_hf_config):
            cfg = Config(
                model="/target", draft="/draft", speculate=True,
                draft_async=True, jit_speculate=True, speculate_k=20,
                max_model_len=4096, extend_draft_rope=True,
                duet_enabled=True, duet_only_proxy=True,
                duet_exit_layer=56, duet_phase1_k=10, duet_phase2_k=10,
                duet_draft_fan_out=1, duet_p2_budget=88,
                duet_p1_tree_policy="off", duet_p2_tree_policy="off",
            )
        self.assertEqual(cfg.duet_proxy_total_budget, 88)
        self.assertEqual(cfg.duet_proxy_wire_N, 90)

    def test_only_proxy_rejects_tree_modes(self):
        with patch("ssd.config.os.path.isdir", return_value=True), \
             patch("ssd.config.AutoConfig.from_pretrained",
                   side_effect=_fake_hf_config), \
             self.assertRaisesRegex(ValueError, "chain-only"):
            Config(
                model="/target", draft="/draft", speculate=True,
                draft_async=True, jit_speculate=True, speculate_k=20,
                duet_enabled=True, duet_only_proxy=True,
                duet_exit_layer=56, duet_phase1_k=10, duet_phase2_k=10,
                duet_draft_fan_out=1, duet_p2_budget=22,
                duet_p1_tree_policy="off", duet_p2_tree_policy="on",
            )


class TestOnlyProxyDataPath(unittest.TestCase):
    def test_empty_phase1_axis_disables_dedup(self):
        positions = torch.arange(90, dtype=torch.int64).remainder(11)
        tokens = torch.arange(1000, 1090, dtype=torch.int64)
        proxy = {
            "chosen_pos": positions.unsqueeze(0),
            "chosen_tok": tokens.unsqueeze(0),
        }
        no_phase1 = torch.empty((1, 11, 0), dtype=torch.int64)

        selected, fan_out = \
            DraftRunner._select_proxy_sourced_tokens_unified(
                proxy, no_phase1, K_rank=10, total_budget=88)

        self.assertEqual(selected.shape, (1, 88))
        self.assertEqual(fan_out.shape, (1, 11))
        self.assertEqual(int(fan_out.sum()), 88)
        self.assertEqual(set(selected.flatten().tolist()),
                         set(tokens[:88].tolist()))

    def test_larger_budget_is_nested_for_a_fixed_proxy_wire(self):
        """Only-proxy must add lower-ranked roots, not replace old roots."""
        positions = torch.arange(90, dtype=torch.int64).remainder(11)
        tokens = torch.arange(2000, 2090, dtype=torch.int64)
        proxy = {
            "chosen_pos": positions.unsqueeze(0),
            "chosen_tok": tokens.unsqueeze(0),
        }
        no_phase1 = torch.empty((1, 11, 0), dtype=torch.int64)

        selected_sets = []
        for budget in (55, 77, 88):
            selected, fan_out = \
                DraftRunner._select_proxy_sourced_tokens_unified(
                    proxy, no_phase1, K_rank=10, total_budget=budget)
            self.assertEqual(int(fan_out.sum()), budget)
            selected_sets.append(set(selected.flatten().tolist()))

        self.assertTrue(selected_sets[0] <= selected_sets[1])
        self.assertTrue(selected_sets[1] <= selected_sets[2])

    def test_cache_contains_proxy_rows_only(self):
        runner = DraftRunner.__new__(DraftRunner)
        runner.config = SimpleNamespace(
            duet_phase1_k=10, duet_phase2_k=10, speculate_k=20,
            duet_tree_policy="off")
        runner.device = torch.device("cpu")
        runner._p1_tree_views = None
        runner._tree_views = None

        width, depth, vocab = 22, 10, 7
        proxy_args = {
            "seq_ids_expanded": torch.zeros(width, dtype=torch.int64),
            "rec_flat": torch.arange(width, dtype=torch.int64),
        }
        proxy_layout = SimpleNamespace(
            fan_idx_per_seq=True,
            fan_idx_hit=torch.arange(width, dtype=torch.int64),
            fan_idx_miss=torch.arange(width, dtype=torch.int64),
        )
        proxy_tokens = torch.ones((width, depth), dtype=torch.int64)
        proxy_logits = torch.zeros((width, depth, vocab))

        runner._merge_and_populate_cache(
            None, None, None,
            proxy_args, proxy_tokens, proxy_logits,
            [False], proxy_layout=proxy_layout, draft_layout=None)

        self.assertEqual(runner._last_n_draft_keys, 0)
        self.assertEqual(runner.tree_cache_keys.shape, (width, 3))
        self.assertEqual(runner.tree_cache_tokens.shape, (width, depth))
        self.assertTrue(torch.equal(
            runner.tree_cache_valid_k,
            torch.full((width,), depth, dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
