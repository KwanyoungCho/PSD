"""M3 unit tests for B>1 support (docs/duet/13-b-gt-1-design.md, stage M3).

Tests the REAL (batched) DraftRunner._select_proxy_sourced_tokens_unified
against a per-seq loop of the ORIGINAL pre-M3 single-seq selector (copied
verbatim below as _select_ref_single) for B=1,2,3 random inputs. Covers:

(a) exact per-seq equality of result_tokens and fan_out rows (incl. planted
    dedup collisions against Phase 1 candidates);
(b) short-seq rows (chosen_pos <= vk_i < K_rank): fan_out beyond vk_i is 0;
(c) non-uniform Phase 1 mask path (zero-padded slots + chosen_tok==0
    false-match guard);
(d) the M3 per-seq fan_idx formula (arange.repeat(B).repeat_interleave of
    the flattened [B,P] fan_out) == concat of per-seq pre-M3
    repeat_interleave, incl. zero rows at padded positions (doc risk (a));
(e) the per-seq glue mask blocks (np.repeat(tril, fol_b)) == the pre-M3
    shared build applied to each seq's list.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_b_gt1_m3

(Note: do NOT use `ssd.tests.X` — that loads the ssd package's __init__ which
imports llm.py → paths.py → requires SSD_HF_CACHE / SSD_DATASET_DIR. Loading
via top-level `tests.X` keeps imports lean.)
"""
import os
import unittest

# Env must be set BEFORE importing ssd.engine.draft_runner (module-level
# constants are baked at import time).
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/data2/chokwans99/datasets")
os.environ["SSD_FORCE_SPLIT_K1K2"] = "1"

import numpy as np
import torch

from ssd.engine.draft_runner import DraftRunner


# ---------------------------------------------------------------------------
# ORIGINAL single-seq selector (pre-M3 draft_runner.py, copied verbatim;
# only the debug asserts' framing kept). Applied per seq as the reference.
# ---------------------------------------------------------------------------

def _select_ref_single(duet_proxy, draft_forked, K_rank, total_budget,
                       draft_forked_mask=None):
    """Pre-M3 _select_proxy_sourced_tokens_unified (B==1).

    duet_proxy: {"chosen_pos": [wire_N], "chosen_tok": [wire_N]};
    draft_forked: [1, P, max_fo]. Returns ([1, total_budget], [K_rank+1]).
    """
    chosen_pos = duet_proxy["chosen_pos"]    # [wire_N]
    chosen_tok = duet_proxy["chosen_tok"]    # [wire_N]
    B = draft_forked.shape[0]
    assert B == 1, "DUET invariant: B=1"
    assert (chosen_pos <= K_rank).all().item()

    df_per_cand = draft_forked[0, chosen_pos, :]                # [N, max_fo]
    eq = (df_per_cand == chosen_tok.unsqueeze(-1))              # [N, max_fo]
    if draft_forked_mask is not None:
        msk = draft_forked_mask[chosen_pos, :]                  # [N, max_fo]
        eq = eq & msk
    in_draft = eq.any(-1)                                       # [N]
    valid = ~in_draft                                           # [N]

    rank = valid.to(torch.int64).cumsum(0)                      # [N]
    take = valid & (rank <= total_budget)                       # [N]

    _take_sum = int(take.sum().item())
    assert _take_sum == total_budget, \
        f"reference underfill: {_take_sum} != {total_budget} (fix the test input)"

    taken_pos = chosen_pos[take]                                # [total_budget]
    taken_tok = chosen_tok[take]

    K_plus_1 = K_rank + 1
    fan_out_tensor = torch.zeros(
        K_plus_1, dtype=torch.int64, device=chosen_pos.device)
    fan_out_tensor.scatter_add_(0, taken_pos, torch.ones_like(taken_pos))

    result = torch.zeros(
        B, total_budget, dtype=torch.int64, device=chosen_pos.device)
    order = taken_pos.argsort(stable=True)
    result[0, :taken_tok.shape[0]] = taken_tok[order]
    return result, fan_out_tensor


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

# Champion-like shapes: K1=9, K2=4, dfo=2, pfo=1 → total_budget=6, wire_N=28.
P = 10            # Phase 1 position_count (K1+1)
MAX_FO = 2        # dfo
TOTAL_BUDGET = 6
WIRE_N = 28
DRAFT_TOK_LO, DRAFT_TOK_HI = 1, 1000       # Phase 1 candidate token range
PROXY_TOK_LO, PROXY_TOK_HI = 1000, 32000   # proxy token range (disjoint)


def _build_case(B, K_rank, seed, n_collisions, vk_per_seq=None,
                with_mask=False, fan_out_list=None):
    """Random selector inputs with planted dedup collisions.

    vk_per_seq: optional [B] ints; seq b's chosen_pos drawn from [0, vk_b]
                (short-seq simulation — M1 construction guarantees the proxy
                never lands beyond vk_i).
    with_mask:  non-uniform Phase 1 — fan_out_list gives real slot counts,
                padding slots are zero and mask filters them; some planted
                chosen_tok==0 hit padded slots (must NOT dedup).
    """
    g = torch.Generator().manual_seed(seed)
    draft_forked = torch.randint(
        DRAFT_TOK_LO, DRAFT_TOK_HI, (B, P, MAX_FO), generator=g)
    mask = None
    if with_mask:
        assert fan_out_list is not None and len(fan_out_list) == P
        mask = torch.zeros(P, MAX_FO, dtype=torch.bool)
        for p, fo in enumerate(fan_out_list):
            mask[p, :fo] = True
        draft_forked = draft_forked * mask  # zero-pad the non-real slots

    if vk_per_seq is None:
        chosen_pos = torch.randint(0, K_rank + 1, (B, WIRE_N), generator=g)
    else:
        chosen_pos = torch.stack([
            torch.randint(0, int(vk) + 1, (WIRE_N,), generator=g)
            for vk in vk_per_seq])
    # Proxy tokens disjoint from the draft range -> no accidental dedup;
    # WIRE_N - n_collisions - n_zero_plants >= TOTAL_BUDGET keeps the
    # buffer-sizing invariant intact.
    chosen_tok = torch.randint(
        PROXY_TOK_LO, PROXY_TOK_HI, (B, WIRE_N), generator=g)

    # Plant dedup collisions: candidate n copies a REAL Phase-1 slot at its pos.
    for b in range(B):
        cols = torch.randperm(WIRE_N, generator=g)[:n_collisions]
        for n in cols.tolist():
            p = int(chosen_pos[b, n])
            j = int(torch.randint(0, MAX_FO, (1,), generator=g)) if mask is None \
                else int(torch.randint(0, max(1, int(mask[p].sum())), (1,), generator=g))
            chosen_tok[b, n] = draft_forked[b, p, j]
    if with_mask:
        # Plant chosen_tok==0 on a padded slot's position: without the mask
        # this would false-match the zero padding; with it, it must survive.
        pad_pos = [p for p, fo in enumerate(fan_out_list)
                   if fo < MAX_FO and p <= K_rank]
        if pad_pos:
            for b in range(B):
                chosen_pos[b, 0] = pad_pos[0]
                chosen_tok[b, 0] = 0
    return {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}, \
        draft_forked, mask


# ---------------------------------------------------------------------------
# (a)-(c) Batched selector == per-seq reference loop
# ---------------------------------------------------------------------------

class TestBatchedSelector(unittest.TestCase):

    def _check(self, duet_proxy, draft_forked, K_rank, mask=None):
        B = draft_forked.shape[0]
        res_b, fo_b = DraftRunner._select_proxy_sourced_tokens_unified(
            duet_proxy, draft_forked, K_rank=K_rank,
            total_budget=TOTAL_BUDGET, draft_forked_mask=mask)
        self.assertEqual(list(res_b.shape), [B, TOTAL_BUDGET])
        self.assertEqual(list(fo_b.shape), [B, K_rank + 1])
        self.assertTrue((fo_b.sum(dim=1) == TOTAL_BUDGET).all(),
                        f"fan_out rows must each sum to total_budget: {fo_b}")
        for b in range(B):
            res_r, fo_r = _select_ref_single(
                {"chosen_pos": duet_proxy["chosen_pos"][b],
                 "chosen_tok": duet_proxy["chosen_tok"][b]},
                draft_forked[b:b + 1], K_rank, TOTAL_BUDGET,
                draft_forked_mask=mask)
            self.assertTrue(torch.equal(res_b[b], res_r[0]),
                            f"result mismatch b={b} B={B} K_rank={K_rank}")
            self.assertTrue(torch.equal(fo_b[b], fo_r),
                            f"fan_out mismatch b={b} B={B} K_rank={K_rank}")
        return fo_b

    def test_uniform_b1_b2_b3(self):
        # K_rank ∈ {K2=4, K1=9}; dedup collisions 0 / 5 / 12 planted.
        for B in (1, 2, 3):
            for K_rank in (4, 9):
                for seed, n_coll in ((0, 0), (1, 5), (2, 12)):
                    duet_proxy, df, _ = _build_case(
                        B, K_rank, seed=100 * B + 10 * K_rank + seed,
                        n_collisions=n_coll)
                    self._check(duet_proxy, df, K_rank)

    def test_short_seq_rows(self):
        # Mixed batch: vk = [9, 4, 4] with K_rank = vk_max = 9. Short seqs'
        # chosen_pos <= vk_i by M1 construction; their fan_out rows must be
        # zero beyond vk_i and still match the reference exactly.
        B, K_rank = 3, 9
        vk = [9, 4, 4]
        for seed in (3, 4):
            duet_proxy, df, _ = _build_case(
                B, K_rank, seed=seed, n_collisions=6, vk_per_seq=vk)
            fo_b = self._check(duet_proxy, df, K_rank)
            for b, vk_b in enumerate(vk):
                self.assertTrue((fo_b[b, vk_b + 1:] == 0).all(),
                                f"fan_out beyond vk_i must be 0: b={b} {fo_b[b]}")

    def test_nonuniform_mask_zero_token(self):
        # Champion Phase 1 list 2,2,2,2,2,2,1,1,1,1 (positions 6..9 padded);
        # chosen_tok==0 planted on a padded position must NOT be deduped.
        fol = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
        for B in (1, 3):
            for K_rank in (4, 9):
                duet_proxy, df, mask = _build_case(
                    B, K_rank, seed=50 + B, n_collisions=4,
                    with_mask=True, fan_out_list=fol)
                self._check(duet_proxy, df, K_rank, mask=mask)

    def test_b1_identity_with_m1_wire_shape(self):
        # B=1 wire tensors [1, wire_N] (post-M1 unpack shape): batched output
        # row 0 must equal the reference fed the squeezed 1-D tensors.
        duet_proxy, df, _ = _build_case(1, 9, seed=7, n_collisions=8)
        self._check(duet_proxy, df, 9)


# ---------------------------------------------------------------------------
# (d) per-seq fan_idx formula (== _update_phase2_layout_inplace math)
# ---------------------------------------------------------------------------

class TestPerSeqFanIdx(unittest.TestCase):
    def _batched_fan_idx(self, fan_out_tensor):
        # Mirror of _update_phase2_layout_inplace (M3).
        B, K_plus_1 = fan_out_tensor.shape
        return torch.arange(K_plus_1, dtype=torch.int64).repeat(B) \
            .repeat_interleave(fan_out_tensor.reshape(-1))

    def _ref_fan_idx(self, fan_out_tensor):
        # concat of the pre-M3 per-seq formula.
        B, K_plus_1 = fan_out_tensor.shape
        return torch.cat([
            torch.arange(K_plus_1, dtype=torch.int64)
            .repeat_interleave(fan_out_tensor[b]) for b in range(B)])

    def test_matches_per_seq_reference(self):
        g = torch.Generator().manual_seed(21)
        for B in (1, 2, 3):
            for K_plus_1 in (5, 10):
                for _ in range(5):
                    # Random rows summing to TOTAL_BUDGET (selector invariant).
                    pos = torch.randint(0, K_plus_1, (B, TOTAL_BUDGET),
                                        generator=g)
                    fo = torch.zeros(B, K_plus_1, dtype=torch.int64)
                    fo.scatter_add_(1, pos, torch.ones_like(pos))
                    out = self._batched_fan_idx(fo)
                    self.assertEqual(out.shape[0], B * TOTAL_BUDGET)
                    self.assertTrue(torch.equal(out, self._ref_fan_idx(fo)))

    def test_zero_rows_at_padded_positions(self):
        # Doc risk (a): short seq concentrates all budget at pos<=vk_i, the
        # padded tail rows are 0 — repeat_interleave must stay consistent.
        fo = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                           [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 6]], dtype=torch.int64)
        out = self._batched_fan_idx(fo)
        self.assertTrue(torch.equal(out, self._ref_fan_idx(fo)))
        self.assertEqual(out.shape[0], 3 * TOTAL_BUDGET)


# ---------------------------------------------------------------------------
# (e) per-seq glue mask blocks (== run_fi_tree_decode_cudagraph M3 branch)
# ---------------------------------------------------------------------------

class TestPerSeqGlueMask(unittest.TestCase):
    def test_nested_build_equals_flat_per_seq(self):
        K_for_mask = 9
        _tril = np.tril(np.ones((K_for_mask + 1, K_for_mask + 1),
                                dtype=np.uint8))
        nested = [[1, 2, 0, 0, 1, 0, 0, 1, 0, 1],
                  [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
        per_seq = [np.repeat(_tril, f, axis=0) for f in nested]  # M3 branch
        for b, fol_b in enumerate(nested):
            flat = np.repeat(_tril, fol_b, axis=0)               # pre-M3 build
            self.assertTrue(np.array_equal(per_seq[b], flat))
            self.assertEqual(per_seq[b].shape,
                             (sum(fol_b), K_for_mask + 1))


if __name__ == "__main__":
    unittest.main()
