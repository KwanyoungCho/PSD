"""Unit tests for Policy B unified selector with padded/mask path.

Tests the non-uniform Phase 1 fan-out support:
- padded false-match guard (chosen_tok=0, mask=False slot must NOT match)
- uniform-list parity (mask=None vs mask=all-True)
- shape contracts

Run: python -m unittest tests.test_policy_b_unified_padded
"""
import os
import unittest

# Skip CUDA-aware imports / SSD config init for unit isolation. We only need
# the selector function which is a method on DraftRunner. Load it without
# triggering full draft init by extracting the method into a thin wrapper.
import torch


def _selector_padded(mesa_proxy, draft_forked, K_rank, total_budget,
                      draft_forked_mask=None):
    """Standalone copy of the dedup core in
    DraftRunner._select_proxy_sourced_tokens_unified, for unit testing.

    Mirrors the exact dedup logic. If this drifts, update both.
    """
    chosen_pos = mesa_proxy["chosen_pos"]
    chosen_tok = mesa_proxy["chosen_tok"]
    N = chosen_pos.shape[0]
    B = draft_forked.shape[0]
    assert B == 1
    df_per_cand = draft_forked[0, chosen_pos, :]
    eq = (df_per_cand == chosen_tok.unsqueeze(-1))
    if draft_forked_mask is not None:
        msk = draft_forked_mask[chosen_pos, :]
        eq = eq & msk
    in_draft = eq.any(-1)
    valid = ~in_draft
    rank = valid.to(torch.int64).cumsum(0)
    take = valid & (rank <= total_budget)
    return take, in_draft


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
        # padded slot.
        chosen_pos = torch.tensor([1, 2, 0, 0])    # try all positions
        chosen_tok = torch.tensor([0, 0, 99, 50])  # 0 first two, then real-match, then no-match
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        take, in_draft = _selector_padded(
            mesa_proxy, padded, K_rank=2, total_budget=4,
            draft_forked_mask=mask)

        # Without mask, chosen_tok=0 at pos 1 (all padding) would match → wrong.
        # With mask, no match.
        self.assertFalse(in_draft[0].item(),
            "chosen_tok=0 at all-padded position 1 must NOT match (mask=False)")
        self.assertFalse(in_draft[1].item(),
            "chosen_tok=0 at position 2 padded slot must NOT match (mask=False there)")
        # Real overlap at pos 0 with tok=99 must match
        self.assertTrue(in_draft[2].item(),
            "chosen_tok=99 at position 0 (real slot) must match")
        # No-match at pos 0 with tok=50 (not in [99, 88])
        self.assertFalse(in_draft[3].item(),
            "chosen_tok=50 at position 0 (no real match) must NOT match")

    def test_no_mask_means_all_real_uniform_path(self):
        # Uniform Phase 1: padded already full, mask=None signals all-real.
        # All slots count as real, including the value 0 if it's actually drafted.
        P, dfo = 3, 2
        # All slots are real. dfo=2 per pos.
        padded = torch.tensor(
            [[[10, 20], [30, 40], [50, 60]]], dtype=torch.int64)

        # chosen_tok=10 at pos 0 must match; chosen_tok=99 at pos 0 must not.
        chosen_pos = torch.tensor([0, 1, 2])
        chosen_tok = torch.tensor([10, 99, 50])
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        take, in_draft = _selector_padded(
            mesa_proxy, padded, K_rank=2, total_budget=3,
            draft_forked_mask=None)   # uniform path

        self.assertTrue(in_draft[0].item(),
            "tok=10 in pos 0 [10, 20] must match")
        self.assertFalse(in_draft[1].item(),
            "tok=99 in pos 1 [30, 40] must NOT match")
        self.assertTrue(in_draft[2].item(),
            "tok=50 in pos 2 [50, 60] must match")

    def test_chosen_tok_zero_matches_real_zero_with_mask(self):
        # Edge case: an actual real slot has token 0 (unlikely but possible).
        # Mask says it's real → chosen_tok=0 should match here.
        P, max_fo = 2, 2
        # padded[0, 0] = [0, 5] — token 0 is REAL at pos 0 slot 0
        # padded[0, 1] = [7, 0] — slot 1 is padding (mask False)
        padded = torch.tensor([[[0, 5], [7, 0]]], dtype=torch.int64)
        mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

        chosen_pos = torch.tensor([0, 1])
        chosen_tok = torch.tensor([0, 0])
        mesa_proxy = {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

        take, in_draft = _selector_padded(
            mesa_proxy, padded, K_rank=1, total_budget=2,
            draft_forked_mask=mask)

        self.assertTrue(in_draft[0].item(),
            "chosen_tok=0 at pos 0 slot 0 (real, value 0) must match")
        self.assertFalse(in_draft[1].item(),
            "chosen_tok=0 at pos 1 must NOT match (slot 0 is 7, slot 1 is padding)")


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

        take_a, indraft_a = _selector_padded(
            mesa_proxy, padded, K_rank=3, total_budget=6,
            draft_forked_mask=None)
        take_b, indraft_b = _selector_padded(
            mesa_proxy, padded, K_rank=3, total_budget=6,
            draft_forked_mask=mask)
        self.assertTrue(torch.equal(indraft_a, indraft_b),
            "all-True mask should yield same result as mask=None")
        self.assertTrue(torch.equal(take_a, take_b))


if __name__ == "__main__":
    unittest.main()
