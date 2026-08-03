"""M6 diagnostic + regression tests for the B>1 short-row collapse
(docs/duet/13-b-gt-1-design.md, stage M6).

Root cause (found 2026-07-18): the target verify input window is built
uniformly as ``pos0 = seq.num_tokens - (vk_max + 1)``
(runner_helpers.prepare_decode_tensors_from_seqs, k = _duet_step_lookahead
= vk_max), but pre-M6 the speculator extended each seq's token_ids by its
OWN vk_i. A short row (vk_i < vk_max) in a mixed batch therefore had pos0
slide back by (vk_max - vk_i) tokens into already-known context — every
logits_p row of that seq was shifted, its accepted length collapsed to ~0
and its recovery token was sampled from a stale position. Impossible at
B=1 (the single seq's vk_i IS vk_max); the guarding
``num_cached_tokens == pos0`` assert is stripped under ``python -O``.

Tests:

(1) FULL DRAFT-SIDE CHAIN (suspects 1-5 of the audit — all clean): REAL
    _unpack_duet_proxy (wire order) → REAL batched selector → REAL
    _update_phase2_layout_inplace → REAL _build_tree_decode_args_for_layout
    → REAL _merge_and_populate_cache, B=3 distinct per-seq data. Every row's
    (seq, k_idx, position, rope, seed) tuple must equal the B=1 run of the
    same seq.
(2) THE BUG (suspect 6): REAL extend_seqs_for_verify + REAL
    prepare_decode_tensors_from_seqs (GPU-gated) on a mixed batch — window
    == [rec] + spec[:vk_max] per seq, pos0 == num_cached_tokens. A
    companion test emulates the PRE-M6 per-seq extension and shows the
    shifted window + that prepare's own assert catches it (without -O).
(3) verify(): per-seq clamp caps a short row at vk_i even when padded
    columns would ratio-accept; recovery comes from logits_p[b, vk_i].
(4) proxy h-padding (M6 in _compute_and_send_proxy): a short seq's chosen
    positions never land beyond vk_i; long rows bit-identical to the
    valid_k=None path.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_b_gt1_m6_verify_window
"""
import os
import unittest
from types import SimpleNamespace

# Env must be set BEFORE importing ssd modules (module constants baked at
# import time).
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/data2/chokwans99/datasets")
os.environ["SSD_FORCE_SPLIT_K1K2"] = "1"

import torch

from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.tree_layout import create_tree_layout
from ssd.utils.verify import verify

# Champion shapes: K1=9, K2=4, dfo=2, pfo=1 → P=10, budget=10, wire_N=28.
K1, K2 = 9, 4
P = K1 + 1
MAX_FO = 2
FOL_P1 = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1]     # champion phase-1 list, MQ1=16
TOTAL_BUDGET = 10                            # pfo * (K_max+1)
WIRE_N = 28
DEV = torch.device("cpu")


def _mk_stub_runner():
    """Minimal DraftRunner stand-in for the layout/build/merge methods."""
    stub = SimpleNamespace()
    stub.device = DEV
    stub.config = SimpleNamespace(
        use_eagle=False,
        duet_proxy_on_draft=False,
        duet_proxy_wire_N=WIRE_N,
        duet_phase1_k=K1,
        duet_phase2_k=K2,
        speculate_k=K1 + K2,
    )
    stub.split_k2_layout = create_tree_layout(
        "split_k2", [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], K=K2, device=DEV)
    return stub


def _mk_chain_inputs(B, vk, seed=0):
    """Distinct per-seq selector inputs. Seq b's proxy tokens live in
    [10000*(b+1), 10000*(b+1)+5000) so any cross-seq leak is detectable."""
    g = torch.Generator().manual_seed(seed)
    draft_forked = torch.randint(1, 1000, (B, P, MAX_FO), generator=g)
    mask = torch.zeros(P, MAX_FO, dtype=torch.bool)
    for p, fo in enumerate(FOL_P1):
        mask[p, :fo] = True
    draft_forked = draft_forked * mask
    chosen_pos = torch.stack([
        torch.randint(0, int(vk[b]) + 1, (WIRE_N,), generator=g)
        for b in range(B)])
    chosen_tok = torch.stack([
        torch.randint(10000 * (b + 1), 10000 * (b + 1) + 5000, (WIRE_N,),
                      generator=g)
        for b in range(B)])
    return ({"chosen_pos": chosen_pos, "chosen_tok": chosen_tok},
            draft_forked, mask)


def _run_chain(stub, duet_proxy, draft_forked, mask, K_rank, num_tokens,
               seq_ids):
    """selector → layout update → build args, with the REAL methods."""
    B = draft_forked.shape[0]
    proxy_forked, fo_t = DraftRunner._select_proxy_sourced_tokens_unified(
        duet_proxy, draft_forked, K_rank=K_rank, total_budget=TOTAL_BUDGET,
        draft_forked_mask=mask)
    layout = DraftRunner._update_phase2_layout_inplace(stub, fo_t, K_rank)
    partial = {
        "num_tokens": num_tokens,
        "temperatures": torch.full((B,), 0.7),
        "dbt": torch.zeros(B, 4, dtype=torch.int32),
        "seq_ids": seq_ids,
        "cache_hits": torch.ones(B, dtype=torch.int64),
    }
    args = DraftRunner._build_tree_decode_args_for_layout(
        stub, partial, proxy_forked, layout, [1] * B)
    return proxy_forked, fo_t, layout, args


class TestFullDraftChain(unittest.TestCase):
    """(1) suspects 1-5: B=3 chain rows ≡ B=1 runs, tuple by tuple."""

    def test_wire_pack_unpack_roundtrip(self):
        # Verifier packs chosen_pos.reshape(-1) then chosen_tok.reshape(-1);
        # REAL _unpack_duet_proxy must recover the per-seq rows.
        B = 3
        duet_proxy, _, _ = _mk_chain_inputs(B, vk=[9, 9, 9], seed=11)
        buf = torch.cat([duet_proxy["chosen_pos"].reshape(-1),
                         duet_proxy["chosen_tok"].reshape(-1)])
        stub = _mk_stub_runner()
        out = DraftRunner._unpack_duet_proxy(stub, buf, B, K1)
        self.assertTrue(torch.equal(out["chosen_pos"],
                                    duet_proxy["chosen_pos"]))
        self.assertTrue(torch.equal(out["chosen_tok"],
                                    duet_proxy["chosen_tok"]))

    def test_chain_rows_match_b1_runs(self):
        B = 3
        vk = [9, 4, 4]                      # mixed batch, K_rank = vk_max
        K_rank = max(vk)
        num_tokens = torch.tensor([100, 200, 300], dtype=torch.int64)
        seq_ids = torch.tensor([7, 8, 9], dtype=torch.int64)
        duet_proxy, draft_forked, mask = _mk_chain_inputs(B, vk, seed=23)

        stub = _mk_stub_runner()
        pf_b, fo_b, layout_b, args_b = _run_chain(
            stub, duet_proxy, draft_forked, mask, K_rank, num_tokens, seq_ids)
        MQ = TOTAL_BUDGET

        for b in range(B):
            stub1 = _mk_stub_runner()
            pf_1, fo_1, layout_1, args_1 = _run_chain(
                stub1,
                {"chosen_pos": duet_proxy["chosen_pos"][b:b + 1],
                 "chosen_tok": duet_proxy["chosen_tok"][b:b + 1]},
                draft_forked[b:b + 1], mask, K_rank,
                num_tokens[b:b + 1], seq_ids[b:b + 1])
            sl = slice(b * MQ, (b + 1) * MQ)
            # seed tokens (suspect 4: seed input ordering)
            self.assertTrue(torch.equal(args_b["input_ids"][sl],
                                        args_1["input_ids"]),
                            f"seed rows differ for seq {b}")
            # fan_out / fan_idx (suspects 2, 3)
            self.assertTrue(torch.equal(fo_b[b], fo_1[0]))
            self.assertTrue(torch.equal(layout_b.fan_idx_hit[sl],
                                        layout_1.fan_idx_hit))
            # positions / rope (suspect 3: per-seq num_tokens alignment)
            self.assertTrue(torch.equal(args_b["positions"][sl],
                                        args_1["positions"]),
                            f"positions differ for seq {b}")
            self.assertTrue(torch.equal(args_b["rope_positions"][sl],
                                        args_1["rope_positions"]),
                            f"rope differs for seq {b}")
            # seq ids
            self.assertTrue((args_b["seq_ids_expanded"][sl]
                             == seq_ids[b]).all())
            # short seqs: no rows forked past vk_i
            self.assertTrue((fo_b[b, vk[b] + 1:] == 0).all())

    def test_merge_proxy_keys_per_seq(self):
        # (suspect 5): proxy_k cache keys = (seq_id, per-seq fan_idx, seed).
        B = 3
        vk = [9, 4, 4]
        K_rank = max(vk)
        num_tokens = torch.tensor([100, 200, 300], dtype=torch.int64)
        seq_ids = torch.tensor([3, 4, 5], dtype=torch.int64)
        duet_proxy, draft_forked, mask = _mk_chain_inputs(B, vk, seed=31)
        stub = _mk_stub_runner()
        pf_b, fo_b, layout_k2, proxy_args = _run_chain(
            stub, duet_proxy, draft_forked, mask, K_rank, num_tokens, seq_ids)

        layout_k1 = create_tree_layout(
            "split_k1_long", FOL_P1, FOL_P1, K=K1, device=DEV)
        MQ1 = layout_k1.MQ_LEN
        draft_forked_k1 = torch.randint(1, 1000, (B, P, MAX_FO)) \
            .masked_fill(~mask, 0)
        # phase-1 flat seeds per layout order (pos-major, real slots only)
        p1_seeds = torch.cat([
            torch.cat([draft_forked_k1[b, p, :fo]
                       for p, fo in enumerate(FOL_P1)])
            for b in range(B)])
        partial = {
            "num_tokens": num_tokens,
            "temperatures": torch.full((B,), 0.7),
            "dbt": torch.zeros(B, 4, dtype=torch.int32),
            "seq_ids": seq_ids,
            "cache_hits": torch.ones(B, dtype=torch.int64),
        }
        draft_args = DraftRunner._build_tree_decode_args_for_layout(
            stub, partial, p1_seeds.view(B, MQ1), layout_k1, [1] * B)

        V = 32
        DraftRunner._merge_and_populate_cache(
            stub,
            draft_args, torch.zeros(B * MQ1, K1, dtype=torch.int64),
            torch.zeros(B * MQ1, K1, V),
            proxy_args, torch.zeros(B * TOTAL_BUDGET, K2, dtype=torch.int64),
            torch.zeros(B * TOTAL_BUDGET, K2, V),
            [1] * B, None, None,
            proxy_layout=layout_k2, draft_layout=layout_k1)

        n_draft = B * MQ1
        keys = stub.tree_cache_keys
        self.assertEqual(keys.shape[0], n_draft + B * TOTAL_BUDGET)
        proxy_keys = keys[n_draft:]
        for b in range(B):
            sl = slice(b * TOTAL_BUDGET, (b + 1) * TOTAL_BUDGET)
            expect_fi = torch.arange(K_rank + 1).repeat_interleave(fo_b[b])
            self.assertTrue((proxy_keys[sl, 0] == seq_ids[b]).all(),
                            f"proxy key seq_id leak for seq {b}")
            self.assertTrue(torch.equal(proxy_keys[sl, 1], expect_fi),
                            f"proxy key k_idx wrong for seq {b}")
            self.assertTrue(torch.equal(proxy_keys[sl, 2], pf_b[b]),
                            f"proxy key seed wrong for seq {b}")
        # per-row valid_k: draft K1, proxy K2
        self.assertTrue((stub.tree_cache_valid_k[:n_draft] == K1).all())
        self.assertTrue((stub.tree_cache_valid_k[n_draft:] == K2).all())


@unittest.skipUnless(torch.cuda.is_available(), "prepare_* needs CUDA")
class TestVerifyWindow(unittest.TestCase):
    """(2) THE BUG: the target verify window for mixed batches."""

    K_LONG = K1 + K2      # speculate_k buffer width
    BLOCK = 16

    def _mk_seqs(self, vk):
        from ssd.engine.sequence import Sequence
        seqs, specs, recs = [], [], []
        for i, _vk in enumerate(vk):
            base = list(range(1000 * (i + 1), 1000 * (i + 1) + 40))
            seq = Sequence(base)
            rec = 500 + i
            seq.recovery_token_id = rec
            seq.append_token(rec)                     # speculate() step 1
            L0 = len(base)                            # tokens with target KV
            seq.num_cached_tokens = L0
            seq.block_table = list(range(10 * i, 10 * i + 8))
            seqs.append(seq)
            specs.append([2000 * (i + 1) + j for j in range(self.K_LONG)])
            recs.append(rec)
        spec_t = torch.tensor(specs, dtype=torch.int64)
        return seqs, spec_t, recs

    def test_mixed_batch_window_aligned_post_fix(self):
        from ssd.engine.speculator_async import extend_seqs_for_verify
        from ssd.engine.helpers.runner_helpers import (
            prepare_decode_tensors_from_seqs)
        vk = [9, 4, 4]
        vk_max = max(vk)
        seqs, spec_t, recs = self._mk_seqs(vk)
        valid_k = torch.tensor(vk, dtype=torch.int64)
        got_vk_max = extend_seqs_for_verify(seqs, spec_t, valid_k,
                                            self.K_LONG)
        self.assertEqual(got_vk_max, vk_max)
        input_ids, positions, _, _ = prepare_decode_tensors_from_seqs(
            seqs, self.BLOCK, False, True, vk_max)   # internal assert armed
        input_ids = input_ids.view(len(vk), vk_max + 1).cpu()
        positions = positions.view(len(vk), vk_max + 1).cpu()
        for b in range(len(vk)):
            L0 = seqs[b].num_tokens - 1 - vk_max
            expect = [recs[b]] + spec_t[b, :vk_max].tolist()
            self.assertEqual(input_ids[b].tolist(), expect,
                             f"verify window misaligned for seq {b}")
            self.assertEqual(positions[b, 0].item(), L0)
            self.assertEqual(seqs[b].num_cached_tokens, L0)

    def test_pre_m6_extension_slides_short_row_window(self):
        # Documentation of the bug: the OLD per-seq extension shifts a short
        # row's window back by vk_max - vk_i, and prepare's own
        # num_cached_tokens == pos0 assert catches it (when not under -O).
        from ssd.engine.helpers.runner_helpers import (
            prepare_decode_tensors_from_seqs)
        vk = [9, 4]
        vk_max = max(vk)
        seqs, spec_t, recs = self._mk_seqs(vk)
        for i, seq in enumerate(seqs):                # pre-M6 loop
            seq.token_ids.extend(spec_t[i, :vk[i]].tolist())
            seq.num_tokens = len(seq.token_ids)
        # short seq: pos0 = num_tokens-(vk_max+1) = L0 - (vk_max - vk_i)
        L0 = seqs[1].num_cached_tokens
        pos0_short = seqs[1].num_tokens - (vk_max + 1)
        self.assertEqual(pos0_short, L0 - (vk_max - vk[1]))
        window = seqs[1][pos0_short:]
        # window starts with STALE context tokens, not the recovery token
        self.assertNotEqual(window[0], recs[1])
        self.assertEqual(window[:vk_max - vk[1]],
                         seqs[1].token_ids[pos0_short:pos0_short
                                           + (vk_max - vk[1])])
        with self.assertRaises(AssertionError):
            prepare_decode_tensors_from_seqs(seqs, self.BLOCK, False, True,
                                             vk_max)


class TestVerifyClampAndRecovery(unittest.TestCase):
    """(3) verify(): mixed batch clamp + recovery position."""

    def test_clamp_and_recovery_positions(self):
        torch.manual_seed(0)
        V = 64
        B, K = 3, 9
        vk = torch.tensor([9, 4, 4], dtype=torch.int64)
        REC = [40, 0, 42]     # expected recovery per row (row1: pad token 0)
        spec = torch.zeros(B, K + 1, dtype=torch.int64)
        logits_q = torch.zeros(B, K, V)
        logits_p = torch.zeros(B, K + 1, V)
        for b in range(B):
            spec[b, 0] = 30 + b                       # prev recovery
            for j in range(int(vk[b])):
                t = 10 + b + j % 5
                spec[b, j + 1] = t
                logits_q[b, j, t] = 50.0              # q = delta at spec tok
                logits_p[b, j, t] = 50.0              # p = delta → accept
        # row 0 (long): full accept → recovery from p[9]
        logits_p[0, K, REC[0]] = 50.0
        # row 1 (short): padded cols 4..8 would ratio-ACCEPT (p puts delta on
        # the pad token 0, q there is uniform-from-zero-logits) — the per-seq
        # clamp must still cap at vk=4; recovery from p[4] = delta at 0.
        for j in range(4, K + 1):
            logits_p[1, j, 0] = 50.0
        # row 2 (short): real rejection at col 1 → residual recovery REC[2]
        logits_p[2, 1, :] = 0.0
        logits_p[2, 1, REC[2]] = 50.0
        temps = torch.full((B,), 0.7)
        suffixes, recs = verify(
            logits_p, logits_q, spec, temps, temps,
            cache_hits=torch.ones(B, dtype=torch.int64),
            jit_speculate=True, valid_k=vk)
        lens = [len(s) - 1 for s in suffixes]         # accepted (minus rec0)
        self.assertEqual(lens, [9, 4, 1],
                         "per-seq accept clamp broken")
        self.assertEqual(recs, REC, "recovery positions broken")


class TestProxyHPadding(unittest.TestCase):
    """(4) _compute_and_send_proxy: short seq's chosen never beyond vk_i."""

    def _run_proxy(self, valid_k):
        from ssd.engine.verifier import Verifier
        import ssd.utils.async_helpers.nccl_pack as nccl_pack
        torch.manual_seed(7)
        V, B, K = 500, 2, 9
        top_k = 6                                     # (K2+1)*6 = 30 >= 28
        cfg = SimpleNamespace(
            duet_proxy_top_k=top_k, duet_proxy_on_draft=False,
            duet_exit_replica=False, jit_speculate=True,
            duet_proxy_fan_out=1, duet_policy="b",
            duet_proxy_wire_N=WIRE_N, max_num_seqs=4,
            # Tier-3 (docs/duet/16): verifier reads the per-step budget via
            # this helper; the stub mirrors the default-path formula.
            duet_p2_budget_at=lambda K: 1 * (K + 1))
        stub = SimpleNamespace(
            target_model_runner=SimpleNamespace(config=cfg),
            _proxy_send_ring=None, _proxy_send_call_count=0)
        exit_logits = torch.randn(B * (K + 1), V)
        logits_q = torch.randn(B, K, V)
        draft_tokens = torch.randint(1, V, (B, K))
        # Shape α̂ high but STRICTLY < 1 at each row's REAL columns: boost q
        # at the draft token, then copy the q row into the exit row minus
        # 0.1 at the token. pE(y)/pD(y) = e^-0.1·Zq/Zp < 1 exactly (the
        # normalizer shrinks less than the numerator), giving α̂ ≈ 0.97:
        # h > 0 at every real position (post-fix topk containment over ≥
        # wire_N strictly-positive entries within [0, vk_i] is
        # deterministic) AND cumprod ≈ 0.9 survives to the padded columns —
        # pre-M6 their bogus α̂ leaked that mass to h[vk_i+1..] and drew
        # chosen beyond vk_i (the regression this test guards). Applied
        # identically in both runs (long rows comparable).
        _vk_ref = [K, 4]
        for b in range(B):
            for j in range(_vk_ref[b]):
                y = draft_tokens[b, j]
                logits_q[b, j, y] += 8.0
                exit_logits[b * (K + 1) + j] = logits_q[b, j]
                exit_logits[b * (K + 1) + j, y] -= 0.1
        if valid_k is not None:
            # realistic padding: zero q logits + pad token 0 beyond vk_i
            for b in range(B):
                vki = int(valid_k[b])
                logits_q[b, vki:] = 0.0
                draft_tokens[b, vki:] = 0
        sent = {}
        orig = nccl_pack.send_int64
        nccl_pack.send_int64 = (
            lambda pg, dst, pos, tok: sent.update(pos=pos.clone(),
                                                  tok=tok.clone()))
        try:
            Verifier._compute_and_send_proxy(
                stub, exit_logits, draft_tokens, logits_q, B, K,
                None, 0, cache_hits=torch.ones(B, dtype=torch.int64),
                valid_k=valid_k)
        finally:
            nccl_pack.send_int64 = orig
        return (sent["pos"].view(B, WIRE_N), sent["tok"].view(B, WIRE_N))

    def test_short_seq_chosen_within_vk(self):
        vk = torch.tensor([9, 4], dtype=torch.int64)
        pos, _ = self._run_proxy(vk)
        self.assertLessEqual(int(pos[1].max()), 4,
                             "short seq chosen_pos leaked beyond vk_i "
                             "(h padding not applied)")
        self.assertLessEqual(int(pos[0].max()), 9)

    def test_long_rows_unchanged_vs_no_valid_k(self):
        pos_none, tok_none = self._run_proxy(None)
        vk = torch.tensor([9, 4], dtype=torch.int64)
        pos_vk, tok_vk = self._run_proxy(vk)
        self.assertTrue(torch.equal(pos_none[0], pos_vk[0]))
        self.assertTrue(torch.equal(tok_none[0], tok_vk[0]))


if __name__ == "__main__":
    unittest.main()
