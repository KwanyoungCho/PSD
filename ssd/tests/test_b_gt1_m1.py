"""M1 unit tests for B>1 support (docs/duet/13-b-gt-1-design.md, stage M1).

(a) Batched Policy B h/P_iv/chosen math (verifier._compute_and_send_proxy)
    vs a single-seq reference loop for random inputs at B=1,2,3. The
    reference mirrors the pre-M1 verifier code (accept_probs[0] scope);
    the batched mirror copies the M1 verifier lines. Upstream tensors
    (p_E/p_D/residual topk → accept_probs, correction_topk_*) were already
    batched pre-M1 and are unchanged, so the test feeds those as inputs.
(b) verify() per-seq valid_k acceptance clamp
    (accept_until = min(accept_until, valid_k)).

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_b_gt1_m1

(Note: do NOT use `ssd.tests.X` — that loads the ssd package's __init__ which
imports llm.py → paths.py → requires SSD_HF_CACHE / SSD_DATASET_DIR. Loading
via top-level `tests.X` keeps imports lean.)
"""
import os
import unittest

# Set required env before any ssd-internal import that may transitively pull
# paths.py. We don't load any HF model here.
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/data2/chokwans99/datasets")

import torch

from ssd.utils.verify import verify


# ---------------------------------------------------------------------------
# (a) Policy B math mirrors
# ---------------------------------------------------------------------------

def _policy_b_batched(accept_probs, corr_probs, corr_ids, wire_N):
    """Mirror of the M1 batched verifier math (verifier.py Policy B block).

    accept_probs: [B, K]; corr_probs/corr_ids: [B, K+1, top_k].
    Returns h [B, K+1], chosen_pos [B, wire_N], chosen_tok [B, wire_N].
    """
    B, K = accept_probs.shape
    top_k = corr_probs.shape[-1]
    cumprod = torch.cumprod(accept_probs, dim=1)     # [B, K]
    h = torch.zeros(B, K + 1)
    h[:, 0] = 1 - accept_probs[:, 0]
    if K > 1:
        h[:, 1:K] = cumprod[:, :-1] * (1 - accept_probs[:, 1:])
    h[:, K] = cumprod[:, -1]
    P_iv = h.unsqueeze(-1) * corr_probs              # [B, K+1, top_k]
    _, top_idx = P_iv.flatten(1).topk(wire_N, dim=-1)  # [B, wire_N]
    chosen_pos = top_idx // top_k
    chosen_tok = corr_ids.flatten(1).gather(1, top_idx)
    return h, chosen_pos, chosen_tok


def _policy_b_single_ref(accept_probs_b, corr_probs_b, corr_ids_b, wire_N):
    """Pre-M1 single-seq reference (verifier.py before this change, which
    hard-indexed [0]); applied to ONE sequence.

    accept_probs_b: [K]; corr_probs_b/corr_ids_b: [K+1, top_k].
    """
    K = accept_probs_b.shape[0]
    top_k = corr_probs_b.shape[-1]
    cumprod = torch.cumprod(accept_probs_b, dim=0)   # [K]
    h = torch.zeros(K + 1)
    h[0] = 1 - accept_probs_b[0]
    if K > 1:
        h[1:K] = cumprod[:-1] * (1 - accept_probs_b[1:])
    h[K] = cumprod[-1]
    P_iv = h.view(K + 1, 1) * corr_probs_b           # [K+1, top_k]
    _, top_idx = P_iv.flatten().topk(wire_N)         # [wire_N]
    chosen_pos = top_idx // top_k
    chosen_tok = corr_ids_b.view(-1).gather(0, top_idx)
    return h, chosen_pos, chosen_tok


class TestBatchedPolicyB(unittest.TestCase):
    """Batched h/P_iv/chosen == per-seq reference loop for B=1,2,3."""

    def _run_case(self, B, K, top_k, wire_N, seed):
        g = torch.Generator().manual_seed(seed)
        accept_probs = torch.rand(B, K, generator=g)
        # Strictly positive, continuous probs → P_iv entries distinct w.p. 1
        # (no topk tie ambiguity between batched and per-seq kernels).
        corr_probs = torch.rand(B, K + 1, top_k, generator=g) + 0.01
        corr_ids = torch.randint(0, 32000, (B, K + 1, top_k), generator=g)

        h_b, pos_b, tok_b = _policy_b_batched(
            accept_probs, corr_probs, corr_ids, wire_N)
        self.assertEqual(list(h_b.shape), [B, K + 1])
        self.assertEqual(list(pos_b.shape), [B, wire_N])
        self.assertEqual(list(tok_b.shape), [B, wire_N])

        for b in range(B):
            h_r, pos_r, tok_r = _policy_b_single_ref(
                accept_probs[b], corr_probs[b], corr_ids[b], wire_N)
            torch.testing.assert_close(h_b[b], h_r, rtol=0, atol=0)
            self.assertTrue(torch.equal(pos_b[b], pos_r),
                            f"chosen_pos mismatch b={b} B={B} K={K}")
            self.assertTrue(torch.equal(tok_b[b], tok_r),
                            f"chosen_tok mismatch b={b} B={B} K={K}")
        # Invariant preserved batched: chosen_pos ∈ [0, K].
        self.assertTrue((pos_b >= 0).all() and (pos_b <= K).all())

    def test_b1_b2_b3(self):
        # Champion-like shapes: K ∈ {K2=4, K1=9}, top_k/wire_N per config scale.
        for B in (1, 2, 3):
            for K in (4, 9):
                for seed in (0, 1, 2):
                    top_k = 14
                    wire_N = min(40, (K + 1) * top_k)  # (K+1)*top_k >= wire_N
                    self._run_case(B, K, top_k, wire_N, seed)

    def test_k1_edge(self):
        # K=1 skips the h[:, 1:K] branch entirely.
        self._run_case(3, 1, 8, 10, 7)

    def test_flat_wire_roundtrip(self):
        # Wire layout: chosen_pos.reshape(-1) then chosen_tok.reshape(-1);
        # draft unpack views buf[:B*wire_N] / buf[B*wire_N:2*B*wire_N] as
        # [B, wire_N]. Verify the roundtrip is the identity, incl. B=1.
        for B in (1, 3):
            K, top_k, wire_N = 4, 14, 30
            g = torch.Generator().manual_seed(11)
            accept_probs = torch.rand(B, K, generator=g)
            corr_probs = torch.rand(B, K + 1, top_k, generator=g) + 0.01
            corr_ids = torch.randint(0, 32000, (B, K + 1, top_k), generator=g)
            _, pos, tok = _policy_b_batched(
                accept_probs, corr_probs, corr_ids, wire_N)
            buf = torch.cat([pos.reshape(-1), tok.reshape(-1)])
            self.assertEqual(buf.numel(), 2 * B * wire_N)
            pos_rt = buf[:B * wire_N].view(B, wire_N)
            tok_rt = buf[B * wire_N:2 * B * wire_N].view(B, wire_N)
            self.assertTrue(torch.equal(pos_rt, pos))
            self.assertTrue(torch.equal(tok_rt, tok))


# ---------------------------------------------------------------------------
# (b) verify() valid_k clamp
# ---------------------------------------------------------------------------

def _greedy_setup(B, K, V, draft_tokens, recovery_tokens, mismatch_at=None):
    """Build logits so greedy verify accepts all K draft tokens per row,
    except rows listed in mismatch_at {row: pos} which mismatch at pos.

    Returns (logits_p, logits_q, speculations, temps).
    """
    start = 1  # previous recovery token (speculations[:, 0])
    speculations = torch.full((B, K + 1), start, dtype=torch.int64)
    speculations[:, 1:] = draft_tokens
    logits_p = torch.zeros(B, K + 1, V)
    for b in range(B):
        for j in range(K):
            tgt = draft_tokens[b, j].item()
            if mismatch_at is not None and mismatch_at.get(b) == j:
                tgt = (tgt + 1) % V  # force greedy mismatch
            logits_p[b, j, tgt] = 10.0
        logits_p[b, K, recovery_tokens[b]] = 10.0
    logits_q = torch.zeros(B, K, V)
    temps = torch.zeros(B)
    return logits_p, logits_q, speculations, temps


class TestVerifyValidKClamp(unittest.TestCase):
    def test_clamp_limits_accept(self):
        B, K, V = 3, 4, 13
        g = torch.Generator().manual_seed(3)
        draft_tokens = torch.randint(2, V, (B, K), generator=g)
        rec = [5, 6, 7]
        logits_p, logits_q, speculations, temps = _greedy_setup(
            B, K, V, draft_tokens, rec)

        valid_k = torch.tensor([4, 2, 0], dtype=torch.int64)
        suffixes, _ = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps,
            valid_k=valid_k)
        # suffix = [start] + accepted draft tokens; accepted = min(K, valid_k).
        self.assertEqual([len(s) for s in suffixes], [5, 3, 1])
        for b, n in enumerate([4, 2, 0]):
            self.assertEqual(
                suffixes[b],
                [1] + draft_tokens[b, :n].tolist())

    def test_none_default_unclamped(self):
        B, K, V = 2, 4, 13
        g = torch.Generator().manual_seed(4)
        draft_tokens = torch.randint(2, V, (B, K), generator=g)
        logits_p, logits_q, speculations, temps = _greedy_setup(
            B, K, V, draft_tokens, [5, 6])
        suffixes, recs = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps)
        self.assertEqual([len(s) for s in suffixes], [K + 1, K + 1])
        self.assertEqual(recs, [5, 6])

    def test_clamp_noop_when_valid_k_equals_k(self):
        # B=1-identity argument: valid_k == K rows are untouched (min no-op).
        B, K, V = 2, 4, 13
        g = torch.Generator().manual_seed(5)
        draft_tokens = torch.randint(2, V, (B, K), generator=g)
        logits_p, logits_q, speculations, temps = _greedy_setup(
            B, K, V, draft_tokens, [5, 6], mismatch_at={1: 2})
        ref_suffixes, ref_recs = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps)
        clamped_suffixes, clamped_recs = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps,
            valid_k=torch.full((B,), K, dtype=torch.int64))
        self.assertEqual(ref_suffixes, clamped_suffixes)
        self.assertEqual(ref_recs, clamped_recs)

    def test_clamp_after_greedy_mismatch(self):
        # Mismatch at pos 1 with valid_k=3: accept = min(1, 3) = 1.
        B, K, V = 1, 4, 13
        g = torch.Generator().manual_seed(6)
        draft_tokens = torch.randint(2, V, (B, K), generator=g)
        logits_p, logits_q, speculations, temps = _greedy_setup(
            B, K, V, draft_tokens, [5], mismatch_at={0: 1})
        suffixes, _ = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps,
            valid_k=torch.tensor([3], dtype=torch.int64))
        self.assertEqual(len(suffixes[0]), 2)  # start + 1 accepted

    def test_ratio_path_noop_identity_with_full_valid_k(self):
        # temp>0 ratio path (champion regime): same RNG seed, valid_k=None vs
        # valid_k=K must be bit-identical (clamp adds no RNG consumption).
        B, K, V = 2, 4, 13
        g = torch.Generator().manual_seed(8)
        draft_tokens = torch.randint(2, V, (B, K), generator=g)
        logits_p = torch.randn(B, K + 1, V, generator=g)
        logits_q = torch.randn(B, K, V, generator=g)
        speculations = torch.full((B, K + 1), 1, dtype=torch.int64)
        speculations[:, 1:] = draft_tokens
        temps = torch.full((B,), 0.7)
        cache_hits = torch.ones(B, dtype=torch.int64)

        torch.manual_seed(123)
        ref = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps,
            cache_hits=cache_hits, jit_speculate=True)
        torch.manual_seed(123)
        out = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps,
            cache_hits=cache_hits, jit_speculate=True,
            valid_k=torch.full((B,), K, dtype=torch.int64))
        self.assertEqual(ref, out)

    def test_ratio_path_clamped(self):
        # temp>0 with a short row: accepted count never exceeds valid_k.
        B, K, V = 3, 4, 13
        g = torch.Generator().manual_seed(9)
        draft_tokens = torch.randint(2, V, (B, K), generator=g)
        logits_p = torch.randn(B, K + 1, V, generator=g)
        # q == p → accept_probs = 1 → ratio path accepts all K...
        logits_q = logits_p[:, :K, :].clone()
        speculations = torch.full((B, K + 1), 1, dtype=torch.int64)
        speculations[:, 1:] = draft_tokens
        temps = torch.full((B,), 0.7)
        cache_hits = torch.ones(B, dtype=torch.int64)
        valid_k = torch.tensor([4, 2, 1], dtype=torch.int64)
        torch.manual_seed(321)
        suffixes, _ = verify(
            logits_p=logits_p, logits_q=logits_q, speculations=speculations,
            temperatures_target=temps, temperatures_draft=temps,
            cache_hits=cache_hits, jit_speculate=True, valid_k=valid_k)
        # ...but the clamp caps rows 1/2 at their valid_k.
        for b in range(B):
            self.assertLessEqual(len(suffixes[b]) - 1, int(valid_k[b]))


if __name__ == "__main__":
    unittest.main()
