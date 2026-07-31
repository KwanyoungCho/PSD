"""Unit tests for the subset-JIT A/B gate (SSD_DUET_JIT_SUBSET).

Reuses the CPU stub harness from tests.test_b_gt1_m2 (real
DraftRunner.hit_cache_and_respond, sentinel jit stub). Verifies:

(a) Gate ON, mixed batch: the JIT is invoked with ONLY the miss rows
    (compact [n] views), and the FINAL response content (hit rows =
    cached tokens/logits, miss rows = JIT sentinel up to _jit_K,
    valid_k / phase_source / cache_hits) is identical to the gate-OFF
    JIT-all path. Content-equivalence is the gate's core contract.
(b) Gate ON, all-miss batch: falls through to the full-batch JIT path
    (no gather needed — subset == batch).
(c) Gate ON, all-hit batch: JIT still skipped entirely (unchanged).
(d) Gate ON, B=1: single-seq batches are all-hit or all-miss, so the
    subset branch never engages — B=1 path untouched.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_b_gt1_jit_subset
"""
import unittest

import torch

from tests.test_b_gt1_m2 import (  # noqa: F401 (env setup happens on import)
    JIT_TOK, K1, K2, K_LONG, _call, _make_runner,
)
import ssd.engine.draft_runner as dr_mod

_JIT_K = K2  # SSD_DUET_JIT_SHORT=1 is set by the m2 harness import


def _mixed_keys():
    # Row 0 hits cache row 0 (draft, vk=K1), row 1 misses,
    # row 2 hits cache row 2 (proxy, vk=K2).
    return torch.tensor([
        [0, 0, 100],
        [5, 0, 999],
        [0, 3, 102],
    ], dtype=torch.int64)


class _GateMixin:
    def setUp(self):
        self._saved = dr_mod.DUET_JIT_SUBSET
        dr_mod.DUET_JIT_SUBSET = True

    def tearDown(self):
        dr_mod.DUET_JIT_SUBSET = self._saved


class TestSubsetJitMixed(_GateMixin, unittest.TestCase):
    def test_jit_sees_only_miss_rows(self):
        calls = []
        r = _make_runner(calls)
        _call(r, _mixed_keys())
        self.assertEqual(calls, [1])  # one JIT call, 1 row (the miss)

    def test_content_identical_to_jit_all(self):
        keys = _mixed_keys()

        calls_off = []
        r_off = _make_runner(calls_off)
        dr_mod.DUET_JIT_SUBSET = False
        out_off = _call(r_off, keys)
        dr_mod.DUET_JIT_SUBSET = True

        calls_on = []
        r_on = _make_runner(calls_on)
        out_on = _call(r_on, keys)

        self.assertEqual(calls_off, [3])  # JIT-all: whole batch
        self.assertEqual(calls_on, [1])   # subset: miss row only

        (tok_a, log_a, hits_a, vk_a, ph_a) = self._unpack(out_off)
        (tok_b, log_b, hits_b, vk_b, ph_b) = self._unpack(out_on)

        torch.testing.assert_close(hits_a.to(torch.int64),
                                   hits_b.to(torch.int64))
        torch.testing.assert_close(vk_a, vk_b)
        torch.testing.assert_close(ph_a, ph_b)
        # Hit rows (0, 2): cached content up to the row's valid_k either
        # way (columns beyond valid_k are random init sliced away by the
        # speculator — excluded from the contract, same as miss rows).
        for i, vk in ((0, K1), (2, K2)):
            torch.testing.assert_close(tok_a[i, :vk], tok_b[i, :vk])
            torch.testing.assert_close(log_a[i, :vk], log_b[i, :vk])
        # Miss row (1): JIT sentinel up to _jit_K (beyond that is random
        # init sliced away by valid_k — excluded from the contract).
        torch.testing.assert_close(tok_a[1, :_JIT_K], tok_b[1, :_JIT_K])
        self.assertTrue(bool((tok_b[1, :_JIT_K] == JIT_TOK).all()))
        torch.testing.assert_close(log_a[1, :_JIT_K], log_b[1, :_JIT_K])

    @staticmethod
    def _unpack(out):
        # hit_cache_and_respond returns (tokens, logits, cache_hits,
        # valid_k, phase_source, ...) — order per the engine; unpack by
        # shape to stay robust.
        tok = next(t for t in out if t.dim() == 2 and t.dtype == torch.int64
                   and t.shape[1] == K_LONG)
        log = next(t for t in out if t.dim() == 3)
        vecs = [t for t in out if t is not None and torch.is_tensor(t)
                and t.dim() == 1]
        hits = next(t for t in vecs if t.dtype == torch.bool) \
            if any(t.dtype == torch.bool for t in vecs) else \
            next(t for t in vecs if set(t.unique().tolist()) <= {0, 1})
        vk = next(t for t in vecs if t.dtype == torch.int64
                  and t.max() >= K2 and t.max() <= K1)
        ph = next(t for t in vecs if t.dtype == torch.int64
                  and t.max() <= 2 and t is not vk)
        return tok, log, hits, vk, ph


class TestSubsetJitDegenerate(_GateMixin, unittest.TestCase):
    def test_all_miss_uses_full_path(self):
        calls = []
        r = _make_runner(calls)
        keys = torch.tensor([[7, 0, 900], [8, 1, 901]], dtype=torch.int64)
        _call(r, keys)
        self.assertEqual(calls, [2])  # full-batch JIT (no hits → no subset)

    def test_all_hit_skips_jit(self):
        calls = []
        r = _make_runner(calls)
        keys = torch.tensor([[0, 0, 100], [1, 0, 101]], dtype=torch.int64)
        _call(r, keys)
        self.assertEqual(calls, [])

    def test_b1_miss_full_path(self):
        calls = []
        r = _make_runner(calls)
        _call(r, torch.tensor([[5, 0, 999]], dtype=torch.int64))
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
