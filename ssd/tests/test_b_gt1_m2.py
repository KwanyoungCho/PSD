"""M2 unit tests for B>1 support (docs/duet/13-b-gt-1-design.md, stage M2).

Tests the REAL DraftRunner.hit_cache_and_respond on CPU via a stub instance
(DraftRunner.__new__ + hand-set attributes, jit_speculate stubbed with a
sentinel writer). Covers:

(a) Mixed hit/miss fix (design §2): JIT runs for ALL rows on any miss, THEN
    hit rows are overwritten from the cache — hit rows keep cached
    tokens/logits/valid_k/phase_source, miss rows keep JIT output with the
    JIT default valid_k (K2 under SSD_DUET_JIT_SHORT, else K_max).
(b) vk_max dispatch (design §1): _vk_scalar = max(valid_k) over the batch;
    glue input width follows vk_max; all-short batches dispatch K2.
(c) B=1 identity: all-hit skips JIT, all-miss skips the cache overwrite —
    byte-identical to the pre-M2 path.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_b_gt1_m2

(Note: do NOT use `ssd.tests.X` — that loads the ssd package's __init__ which
imports llm.py → paths.py → requires SSD_HF_CACHE / SSD_DATASET_DIR. Loading
via top-level `tests.X` keeps imports lean.)
"""
import os
import unittest

# Env must be set BEFORE importing ssd.engine.draft_runner: module-level
# constants SPLIT_K1K2_MODE / DUET_JIT_SHORT are baked at import time.
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/data2/chokwans99/datasets")
os.environ["SSD_FORCE_SPLIT_K1K2"] = "1"
os.environ["SSD_DUET_JIT_SHORT"] = "1"

import torch

import ssd.engine.draft_runner as dr_mod


# Champion-like shapes: K1=9 K2=4 (k=13); cache row width = K_max = 9.
K1, K2 = 9, 4
K_MAX = max(K1, K2)
K_LONG = K1 + K2   # speculate_k / out_tokens width
V = 50

JIT_TOK = 7        # sentinel token the stub JIT writes
JIT_LOGIT = 0.5    # sentinel logit value


class _StubConfig:
    duet_phase1_k = K1
    duet_phase2_k = K2
    speculate_k = K_LONG
    use_eagle = False
    verbose = False
    jit_speculate = True
    duet_enabled = True


class _StubHFConfig:
    vocab_size = V
    torch_dtype = torch.float32
    hidden_size = 8


def _make_cache():
    """4 cache rows: [0..1] draft-sourced (vk=K1), [2..3] proxy-sourced (vk=K2).

    Keys are (seq_id, k_idx, rec_token). Token values >= 1000 so they can
    never collide with the JIT sentinel.
    """
    g = torch.Generator().manual_seed(0)
    keys = torch.tensor([
        [0, 0, 100],
        [1, 0, 101],
        [0, 3, 102],
        [2, 1, 103],
    ], dtype=torch.int64)
    tokens = torch.arange(4 * K_MAX, dtype=torch.int64).view(4, K_MAX) + 1000
    logits = torch.randn(4, K_MAX, V, generator=g)
    valid_k = torch.tensor([K1, K1, K2, K2], dtype=torch.int64)
    return keys, tokens, logits, valid_k


def _make_runner(jit_calls, empty_cache=False):
    r = dr_mod.DraftRunner.__new__(dr_mod.DraftRunner)
    r.config = _StubConfig()
    r.hf_config = _StubHFConfig()
    r.device = "cpu"
    if empty_cache:
        r.tree_cache_keys = torch.empty((0, 3), dtype=torch.int64)
        r.tree_cache_tokens = torch.empty((0, K_MAX), dtype=torch.int64)
        r.tree_cache_logits = torch.empty((0, K_MAX, V))
        r.tree_cache_valid_k = torch.empty((0,), dtype=torch.int64)
    else:
        (r.tree_cache_keys, r.tree_cache_tokens,
         r.tree_cache_logits, r.tree_cache_valid_k) = _make_cache()
    r.tree_cache_activations = None
    r._last_n_draft_keys = 2

    def _fake_jit(request_keys, num_tokens, out_logits, out_tokens,
                  temperatures, draft_block_tables,
                  target_recovery_activations=None):
        # Mimics the real batched JIT: writes ALL rows, _jit_K columns deep
        # (K2 under SSD_DUET_JIT_SHORT, else K_max) into the caller's
        # out_tokens/out_logits buffers.
        jit_calls.append(int(request_keys.shape[0]))
        _jit_K = K2 if dr_mod.DUET_JIT_SHORT else K_MAX
        out_tokens[:, :_jit_K] = JIT_TOK
        out_logits[:, :_jit_K, :] = JIT_LOGIT
        return None

    r.jit_speculate = _fake_jit
    return r


def _call(r, request_keys):
    B = request_keys.shape[0]
    num_tokens = torch.full((B,), 64, dtype=torch.int64)
    temperatures = torch.zeros(B)
    dbt = torch.zeros((B, 4), dtype=torch.int32)
    return dr_mod.DraftRunner.hit_cache_and_respond(
        r, request_keys, B, K_LONG, num_tokens, temperatures, dbt)


class TestMixedHitMissFill(unittest.TestCase):
    """Design §2: B=3 hit/miss/hit — JIT all, then cache overwrites hits."""

    def _mixed_keys(self):
        # Row 0 hits cache row 0 (draft, vk=K1), row 1 misses,
        # row 2 hits cache row 2 (proxy, vk=K2).
        return torch.tensor([
            [0, 0, 100],
            [5, 0, 999],
            [0, 3, 102],
        ], dtype=torch.int64)

    def test_mixed_hit_miss_hit(self):
        jit_calls = []
        r = _make_runner(jit_calls)
        (out_tokens, out_logits, glue_ids, cache_hits, out_acts,
         phase_source, valid_k, vk_scalar, _prof) = _call(r, self._mixed_keys())

        # JIT ran ONCE for the full batch (any miss → batched JIT all rows).
        self.assertEqual(jit_calls, [3])
        self.assertEqual(cache_hits.tolist(), [True, False, True])

        # Hit rows keep their CACHED tokens/logits (overwritten after JIT).
        self.assertTrue(torch.equal(out_tokens[0, :K_MAX], r.tree_cache_tokens[0]))
        self.assertTrue(torch.equal(out_tokens[2, :K_MAX], r.tree_cache_tokens[2]))
        self.assertTrue(torch.equal(out_logits[0, :K_MAX], r.tree_cache_logits[0]))
        self.assertTrue(torch.equal(out_logits[2, :K_MAX], r.tree_cache_logits[2]))
        self.assertFalse((out_tokens[0, :K_MAX] == JIT_TOK).any())
        self.assertFalse((out_tokens[2, :K_MAX] == JIT_TOK).any())

        # Miss row keeps JIT output (K2-deep under SSD_DUET_JIT_SHORT).
        self.assertTrue((out_tokens[1, :K2] == JIT_TOK).all())
        self.assertTrue((out_logits[1, :K2] == JIT_LOGIT).all())

        # Per-seq valid_k: hit rows = matched cache row's vk, miss = K2
        # (SSD_DUET_JIT_SHORT default). Stays PER-SEQ on the wire.
        self.assertEqual(valid_k.tolist(), [K1, K2, K2])
        # phase_source: hit rows keep classification (1=draft, 2=proxy),
        # miss row 0.
        self.assertEqual(phase_source.tolist(), [1, 0, 2])

        # Dispatch scalar = vk_max over the batch.
        self.assertEqual(vk_scalar, K1)
        # Glue input: [B, vk_max+1] flattened; col 0 = rec token, then the
        # first vk_max out_tokens columns (short/miss rows ride padding).
        glue = glue_ids.view(3, K_MAX + 1)
        self.assertEqual(glue[:, 0].tolist(), [100, 999, 102])
        self.assertTrue(torch.equal(glue[:, 1:], out_tokens[:, :K_MAX]))

    def test_mixed_jit_long_default(self):
        # Without SSD_DUET_JIT_SHORT the miss default is K_max.
        jit_calls = []
        r = _make_runner(jit_calls)
        saved = dr_mod.DUET_JIT_SHORT
        dr_mod.DUET_JIT_SHORT = False
        try:
            (out_tokens, _out_logits, _glue, cache_hits, _acts,
             valid_phase, valid_k, vk_scalar, _p) = _call(r, self._mixed_keys())
        finally:
            dr_mod.DUET_JIT_SHORT = saved
        self.assertEqual(jit_calls, [3])
        self.assertEqual(valid_k.tolist(), [K1, K_MAX, K2])
        self.assertEqual(vk_scalar, K1)
        # Hit rows still overwritten from cache even with the K_max-deep JIT.
        self.assertTrue(torch.equal(out_tokens[0, :K_MAX], r.tree_cache_tokens[0]))
        self.assertTrue((out_tokens[1, :K_MAX] == JIT_TOK).all())

    def test_all_hit_skips_jit(self):
        jit_calls = []
        r = _make_runner(jit_calls)
        keys = torch.tensor([[0, 0, 100], [0, 3, 102]], dtype=torch.int64)
        (out_tokens, _l, _g, cache_hits, _a,
         phase_source, valid_k, vk_scalar, _p) = _call(r, keys)
        self.assertEqual(jit_calls, [])  # all-hit: JIT skipped (pre-M2 path)
        self.assertEqual(cache_hits.tolist(), [True, True])
        self.assertTrue(torch.equal(out_tokens[0, :K_MAX], r.tree_cache_tokens[0]))
        self.assertTrue(torch.equal(out_tokens[1, :K_MAX], r.tree_cache_tokens[2]))
        self.assertEqual(valid_k.tolist(), [K1, K2])
        self.assertEqual(vk_scalar, K1)

    def test_all_short_batch_dispatches_k2(self):
        # All rows hit proxy (vk=K2) rows → vk_max = K2 → short-bucket
        # dispatch and glue width K2+1.
        jit_calls = []
        r = _make_runner(jit_calls)
        keys = torch.tensor([[0, 3, 102], [2, 1, 103]], dtype=torch.int64)
        (_t, _l, glue_ids, _h, _a, _ps, valid_k, vk_scalar, _p) = _call(r, keys)
        self.assertEqual(valid_k.tolist(), [K2, K2])
        self.assertEqual(vk_scalar, K2)
        self.assertEqual(glue_ids.numel(), 2 * (K2 + 1))

    def test_all_miss_no_overwrite(self):
        jit_calls = []
        r = _make_runner(jit_calls)
        keys = torch.tensor([[7, 0, 900], [8, 0, 901]], dtype=torch.int64)
        (out_tokens, _l, _g, cache_hits, _a,
         phase_source, valid_k, vk_scalar, _p) = _call(r, keys)
        self.assertEqual(jit_calls, [2])
        self.assertEqual(cache_hits.tolist(), [False, False])
        self.assertTrue((out_tokens[:, :K2] == JIT_TOK).all())
        self.assertEqual(valid_k.tolist(), [K2, K2])
        self.assertEqual(phase_source.tolist(), [0, 0])
        self.assertEqual(vk_scalar, K2)

    def test_empty_cache_jits_all(self):
        jit_calls = []
        r = _make_runner(jit_calls, empty_cache=True)
        keys = torch.tensor([[0, 0, 100], [1, 0, 101]], dtype=torch.int64)
        (out_tokens, _l, _g, cache_hits, _a,
         _ps, valid_k, vk_scalar, _p) = _call(r, keys)
        self.assertEqual(jit_calls, [2])
        self.assertEqual(cache_hits.tolist(), [0, 0])
        self.assertTrue((out_tokens[:, :K2] == JIT_TOK).all())
        self.assertEqual(vk_scalar, K2)


class TestB1Identity(unittest.TestCase):
    """B=1 can only be all-hit or all-miss — both must match the pre-M2 path."""

    def test_b1_hit(self):
        jit_calls = []
        r = _make_runner(jit_calls)
        keys = torch.tensor([[0, 0, 100]], dtype=torch.int64)
        (out_tokens, out_logits, glue_ids, cache_hits, _a,
         phase_source, valid_k, vk_scalar, _p) = _call(r, keys)
        self.assertEqual(jit_calls, [])
        self.assertTrue(torch.equal(out_tokens[0, :K_MAX], r.tree_cache_tokens[0]))
        self.assertEqual(valid_k.tolist(), [K1])
        self.assertEqual(vk_scalar, K1)  # == old valid_k[0] capture
        self.assertEqual(phase_source.tolist(), [1])

    def test_b1_miss(self):
        jit_calls = []
        r = _make_runner(jit_calls)
        keys = torch.tensor([[5, 0, 999]], dtype=torch.int64)
        (out_tokens, _l, _g, cache_hits, _a,
         phase_source, valid_k, vk_scalar, _p) = _call(r, keys)
        self.assertEqual(jit_calls, [1])
        self.assertTrue((out_tokens[0, :K2] == JIT_TOK).all())
        self.assertEqual(valid_k.tolist(), [K2])
        self.assertEqual(vk_scalar, K2)  # == old valid_k[0] capture
        self.assertEqual(phase_source.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
