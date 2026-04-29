"""Unit tests for Policy B unified selector with padded/mask path.

Tests the non-uniform Phase 1 fan-out support:
- padded false-match guard (chosen_tok=0, mask=False slot must NOT match)
- uniform-list parity (mask=None vs mask=all-True)
- shape contracts

Calls production DraftRunner._select_proxy_sourced_tokens_unified directly
(staticmethod, no instance needed) so logic drift is caught.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_policy_b_unified_padded

(Note: do NOT use `ssd.tests.X` — that loads the ssd package's __init__ which
imports llm.py → paths.py → requires SSD_HF_CACHE / SSD_DATASET_DIR. Loading
via top-level `tests.X` keeps imports lean.)
"""
import os
import unittest

# Set required env before any ssd-internal import that may transitively pull
# paths.py. We don't actually load any HF model here, just import the class.
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/data2/chokwans99/datasets")

import torch

from ssd.engine.draft_runner import DraftRunner

# Production selector (staticmethod). If signature drifts, tests fail at call.
_unified = DraftRunner._select_proxy_sourced_tokens_unified


class TestPaddedFalseMatch(unittest.TestCase):
    """Padded zero slots must NOT cause false in_draft when chosen_tok=0."""

    def test_zero_chosen_tok_with_zero_padding_no_match(self):
        # K_rank=2 → P=3 positions. Non-uniform fan_out_list = [2, 0, 1].
        # max_fo = 2. Position 1 fully padded, position 2 has 1 real + 1 padded.
        P, max_fo = 3, 2
        # padded[0, 0] = [99, 88]   (real)
        # padded[0, 1] = [0, 0]     (all padding)
        # padded[0, 2] = [77, 0]    (1 real + 1 padding)
        padded = torch.tensor(
            [[[99, 88], [0, 0], [77, 0]]], dtype=torch.int64)
        # mask: position 0 both real, position 1 none, position 2 first real
        mask = torch.tensor(
            [[True, True], [False, False], [True, False]], dtype=torch.bool)

        # chosen_tok includes 0 — must NOT false-match position 1 or position 2's
        # padded slot. wire_N=4 (4 candidates, all kept).
        chosen_pos = torch.tensor([1, 2, 0, 0])    # try all positions
        chosen_tok = torch.tensor([0, 0, 99, 50])  # 0 first two, then real-match, then no-match
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        # total_budget=2 — only first 2 valid take. assert in production code
        # would fire if take.sum() != 2. Test focused on dedup correctness:
        # without mask, in_draft[0]/[1] would be True (false-match on 0).
        # With mask, in_draft[0]/[1] = False, in_draft[2] = True, in_draft[3] = False.
        # take = valid & (rank <= 2):
        # valid    = [True, True, False, True]   (pos 0 cand=99 in draft → False)
        # rank     = [1,    2,    2,     3]
        # take     = [T,    T,    F,     F]      sum = 2 ✓
        result, fan_out_tensor = _unified(
            mesa_proxy, padded, K_rank=2, total_budget=2,
            draft_forked_mask=mask)
        # Result must contain chosen_tok values for valid takes (0 and 0 — at
        # positions 1 and 2). int(fan_out_tensor[1] = 1, int(fan_out_tensor[2] = 1, sum=2.
        self.assertEqual(int(fan_out_tensor.sum().item()), 2)
        self.assertEqual(int(fan_out_tensor[1].item()), 1, "pos 1 (chosen_tok=0, all padding) must be taken")
        self.assertEqual(int(fan_out_tensor[2].item()), 1, "pos 2 (chosen_tok=0, first real=77, slot 1 padding) must be taken")
        self.assertEqual(int(fan_out_tensor[0].item()), 0, "pos 0 takes excluded (in_draft on first cand) or beyond budget")

    def test_no_mask_means_all_real_uniform_path(self):
        # Uniform Phase 1: padded already full, mask=None signals all-real.
        P, dfo = 3, 2
        padded = torch.tensor(
            [[[10, 20], [30, 40], [50, 60]]], dtype=torch.int64)

        # All chosen_tok values overlap with respective draft positions.
        # in_draft = [True, False, True] → valid = [F, T, F] → only pos 1 takes.
        chosen_pos = torch.tensor([0, 1, 2])
        chosen_tok = torch.tensor([10, 99, 50])
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        result, fan_out_tensor = _unified(
            mesa_proxy, padded, K_rank=2, total_budget=1,
            draft_forked_mask=None)   # uniform path
        self.assertEqual(int(fan_out_tensor.sum().item()), 1)
        self.assertEqual(int(fan_out_tensor[1].item()), 1, "pos 1 (chosen_tok=99 not in [30,40]) must be taken")

    def test_chosen_tok_zero_matches_real_zero_with_mask(self):
        # Edge case: an actual real slot has token 0. Mask says it's real
        # → chosen_tok=0 should match (proper in_draft) and be excluded.
        P, max_fo = 2, 2
        # padded[0, 0] = [0, 5] — token 0 is REAL at pos 0 slot 0
        # padded[0, 1] = [7, 0] — slot 1 is padding
        padded = torch.tensor([[[0, 5], [7, 0]]], dtype=torch.int64)
        mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

        # chosen_tok=0 at pos 0: real match (excluded).
        # chosen_tok=0 at pos 1: slot 0=7, slot 1=0 padding → NOT match → take.
        chosen_pos = torch.tensor([0, 1])
        chosen_tok = torch.tensor([0, 0])
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        result, fan_out_tensor = _unified(
            mesa_proxy, padded, K_rank=1, total_budget=1,
            draft_forked_mask=mask)
        self.assertEqual(int(fan_out_tensor.sum().item()), 1)
        self.assertEqual(int(fan_out_tensor[1].item()), 1,
            "pos 1 (chosen_tok=0, real slots [7], padding [0]) must take")
        self.assertEqual(int(fan_out_tensor[0].item()), 0,
            "pos 0 (chosen_tok=0 is real → in_draft → excluded) must NOT take")


class TestParityUniformAndPadded(unittest.TestCase):
    """For uniform fan_out_list expressed as padded with all-True mask, results
    must equal the uniform 3D path with mask=None."""

    def test_uniform_parity(self):
        torch.manual_seed(0)
        B, P, dfo = 1, 4, 3
        padded = torch.randint(1, 1000, (B, P, dfo), dtype=torch.int64)
        mask = torch.ones(P, dfo, dtype=torch.bool)

        chosen_pos = torch.tensor([0, 0, 1, 2, 3, 1])
        chosen_tok = torch.tensor([padded[0, 0, 0].item(),
                                    padded[0, 0, 2].item(),
                                    99,
                                    padded[0, 2, 1].item(),
                                    98,
                                    97])
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        result_a, fan_out_a_tensor = _unified(
            mesa_proxy, padded, K_rank=3, total_budget=3,
            draft_forked_mask=None)
        result_b, fan_out_b_tensor = _unified(
            mesa_proxy, padded, K_rank=3, total_budget=3,
            draft_forked_mask=mask)
        # Same dedup outcome → same fan_out tensor.
        self.assertTrue(torch.equal(fan_out_a_tensor, fan_out_b_tensor),
            "all-True mask should yield same fan_out tensor as mask=None")
        # Same result tensor (token order preserved by stable sort).
        self.assertTrue(torch.equal(result_a, result_b))


if __name__ == "__main__":
    unittest.main()
