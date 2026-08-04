"""T1.2 unit tests — P2-tree 사전 예산 배분 (docs/duet/20).

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_p2_tree_alloc
"""
import unittest

import torch

from ssd.engine.helpers.p2_tree import alloc_fanouts, alloc_root_budgets


class TestRootBudgets(unittest.TestCase):
    def test_sum_and_cap(self):
        piv = torch.tensor([0.5, 0.2, 0.1, 0.05])
        b = alloc_root_budgets(piv, total=40, beta=0.5, cap=8)
        self.assertLessEqual(int(b.sum()), 40)
        self.assertTrue(bool((b <= 8).all()))
        self.assertTrue(bool((b >= 0).all()))
        # cap이 배분을 제한하는 레짐: 상위 root가 cap에 닿는다
        self.assertEqual(int(b[0]), 8)

    def test_beta_zero_uniform(self):
        piv = torch.tensor([0.9, 0.01, 0.01, 0.01])
        b = alloc_root_budgets(piv, total=8, beta=0.0, cap=8)
        self.assertEqual(b.tolist(), [2, 2, 2, 2])   # 균등

    def test_beta_one_proportional(self):
        piv = torch.tensor([0.6, 0.3, 0.1])
        b = alloc_root_budgets(piv, total=10, beta=1.0, cap=10)
        self.assertEqual(int(b.sum()), 10)
        self.assertEqual(b.tolist(), [6, 3, 1])

    def test_deterministic(self):
        piv = torch.tensor([0.3, 0.3, 0.3])
        b1 = alloc_root_budgets(piv, 7, 0.5, 8)
        b2 = alloc_root_budgets(piv, 7, 0.5, 8)
        self.assertEqual(b1.tolist(), b2.tolist())
        # 동률은 낮은 인덱스(높은 rank) 우선
        self.assertEqual(b1.tolist(), [3, 2, 2])

    def test_single_root(self):
        b = alloc_root_budgets(torch.tensor([1.0]), total=40, beta=0.5,
                               cap=8)
        self.assertEqual(b.tolist(), [8])            # cap이 지배

    def test_d10_signature(self):
        # D10: 함수가 자식 '정체'를 입력받을 수 없음 — 시그니처 확인.
        import inspect
        params = list(inspect.signature(alloc_root_budgets).parameters)
        self.assertNotIn("tokens", params)
        self.assertNotIn("child_tokens", params)


class TestFanouts(unittest.TestCase):
    def test_prefix_no_overdraw(self):
        # 리뷰 4차 반례: 같은 root 부모 2, remaining 4, c=3 → 3+1 (초과 금지)
        pri = torch.tensor([2.0, 1.0])
        root = torch.tensor([0, 0])
        rem = torch.tensor([4])
        f = alloc_fanouts(pri, root, rem, c_tensor=3)
        self.assertEqual(f.tolist(), [3, 1])
        self.assertEqual(int(f.sum()), 4)

    def test_priority_order_wins(self):
        pri = torch.tensor([1.0, 5.0])               # 부모1이 상위
        root = torch.tensor([0, 0])
        f = alloc_fanouts(pri, root, torch.tensor([4]), c_tensor=3)
        self.assertEqual(f.tolist(), [1, 3])
        # 동률: 낮은 인덱스 우선 (stable)
        f2 = alloc_fanouts(torch.tensor([1.0, 1.0]), root,
                           torch.tensor([4]), c_tensor=3)
        self.assertEqual(f2.tolist(), [3, 1])

    def test_multi_root_independent(self):
        pri = torch.tensor([1.0, 2.0, 3.0])
        root = torch.tensor([0, 1, 0])
        f = alloc_fanouts(pri, root, torch.tensor([3, 2]), c_tensor=3)
        self.assertEqual(f.tolist(), [0, 2, 3])      # root0: 상위가 3 소진
        self.assertEqual(int(f.sum()), 5)

    def test_exhausted_root_zero(self):
        f = alloc_fanouts(torch.tensor([1.0]), torch.tensor([0]),
                          torch.tensor([0]), c_tensor=3)
        self.assertEqual(f.tolist(), [0])

    def test_input_not_mutated(self):
        rem = torch.tensor([4])
        alloc_fanouts(torch.tensor([1.0, 2.0]), torch.tensor([0, 0]),
                      rem, 3)
        self.assertEqual(rem.tolist(), [4])          # pure (복사 후 소모)


if __name__ == "__main__":
    unittest.main()


class TestWorSampler(unittest.TestCase):
    def _mk(self, B=4, V=64, seed=7):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(B, V, generator=g)
        temps = torch.full((B,), 0.7)
        return logits, temps

    def test_c1_bit_identical_to_sampler(self):
        # 현행 Sampler.forward와 RNG 소비·결과 동일 (fast-path 근거)
        from ssd.engine.helpers.p2_tree import tree_sample_wor
        from ssd.layers.sampler import Sampler
        logits, temps = self._mk()
        torch.manual_seed(123)
        ref = Sampler()(logits.clone(), temps)
        state_ref = torch.get_rng_state()
        torch.manual_seed(123)
        tok, _ = tree_sample_wor(logits.clone(), temps, c_tensor=1)
        state_new = torch.get_rng_state()
        self.assertEqual(ref.tolist(), tok.squeeze(1).tolist())
        self.assertTrue(bool((state_ref == state_new).all()))

    def test_wor_no_duplicates_and_order(self):
        from ssd.engine.helpers.p2_tree import tree_sample_wor
        logits, temps = self._mk(B=8, V=32)
        torch.manual_seed(5)
        tok, raw = tree_sample_wor(logits, temps, c_tensor=4)
        for row in tok.tolist():
            self.assertEqual(len(set(row)), 4)          # 비복원
        self.assertEqual(raw.shape, (8, 4))

    def test_raw_q_is_unrenormalized(self):
        # 결정 ②: raw_q는 원본 q_eff 값 (독립 재계산과 일치)
        from ssd.engine.helpers.p2_tree import tree_sample_wor
        logits, temps = self._mk(B=2, V=16)
        torch.manual_seed(9)
        tok, raw = tree_sample_wor(logits.clone(), temps, c_tensor=3)
        expect = torch.softmax(logits / temps.unsqueeze(1), dim=-1)
        got = expect.gather(1, tok)
        self.assertTrue(torch.allclose(raw, got, atol=1e-6))
        # 형제 합 ≤ 1 (원본 규약의 자연 귀결)
        self.assertTrue(bool((raw.sum(1) <= 1.0 + 1e-6).all()))

    def test_temp0_gated(self):
        from ssd.engine.helpers.p2_tree import tree_sample_wor
        logits, _ = self._mk(B=2)
        with self.assertRaises(ValueError):
            tree_sample_wor(logits, torch.tensor([0.7, 0.0]), 2)


class TestPivPack(unittest.TestCase):
    def test_roundtrip_accuracy(self):
        from ssd.engine.helpers.p2_tree import pack_piv, unpack_piv
        tok = torch.randint(0, 32000, (2, 28), dtype=torch.int64)
        piv = torch.tensor([[10 ** (-6 * torch.rand(1).item())
                             for _ in range(28)] for _ in range(2)])
        packed = pack_piv(tok, piv)
        tok2, piv2 = unpack_piv(packed)
        self.assertEqual(tok.tolist(), tok2.tolist())     # 토큰 무손실
        # log10 양자화 오차 ≤ 반스텝 (6/65535 ≈ 9.2e-5 데케이드)
        err = (piv2.log10() - piv.log10()).abs().max().item()
        self.assertLess(err, 6.0 / 65535)
    def test_packed_differs_from_clean(self):
        # dedup 함정 재현 방지: pack된 값은 원 토큰과 달라야 하고
        # (버전 비트), unpack 없이 비교하면 실패해야 정상
        from ssd.engine.helpers.p2_tree import pack_piv
        tok = torch.tensor([[5, 100]], dtype=torch.int64)
        packed = pack_piv(tok, torch.tensor([[0.5, 0.001]]))
        self.assertFalse(bool((packed == tok).any()))
    def test_version_bit_enforced(self):
        from ssd.engine.helpers.p2_tree import unpack_piv
        with self.assertRaises(ValueError):
            unpack_piv(torch.tensor([[5]], dtype=torch.int64))  # pack 안 됨
    def test_extreme_piv_clamped(self):
        from ssd.engine.helpers.p2_tree import pack_piv, unpack_piv
        tok = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        piv = torch.tensor([[1.0, 1e-9, 0.0]])
        _, piv2 = unpack_piv(pack_piv(tok, piv))
        self.assertAlmostEqual(piv2[0, 0].item(), 1.0, places=3)
        self.assertAlmostEqual(piv2[0, 1].item(), 1e-6, places=8)  # 하한 clamp
        self.assertAlmostEqual(piv2[0, 2].item(), 1e-6, places=8)


class TestRolloutReference(unittest.TestCase):
    def _sample_fn(self, seed=3):
        g = torch.Generator().manual_seed(seed)
        def fn(sel, fan):
            n = len(sel)
            C = 3
            toks = torch.randint(100, 200, (n, C), generator=g)
            raws = torch.rand(n, C, generator=g) * 0.3
            return toks, raws
        return fn

    def _run(self, policy, R=4, W=4, F=3):
        from ssd.engine.helpers.p2_tree import rollout_reference
        piv = torch.tensor([0.5 / (1.6 ** r) for r in range(R)])
        return rollout_reference(
            list(range(10, 10 + R)), piv, root_pos=None, policy=policy,
            W=W, F_total=F, c_tensor=3, nv=6, beta=0.5, depth_cap=4,
            sample_fn=self._sample_fn())

    def test_level_depth_sync(self):
        pool, log = self._run("level")
        # level: forward f에서 평가된 노드는 전부 depth == f
        for f, (sel, fan) in enumerate(log):
            for i in sel:
                self.assertEqual(int(pool.depth[i]), f)

    def test_frontier_mixes_depths_level_cannot(self):
        # frontier의 본질: 한 forward에서 서로 다른 depth가 경쟁 가능
        # (보류된 얕은 root vs 새 자식). R>W로 보류를 강제.
        pool, log = self._run("frontier", R=6, W=3, F=3)
        depth_sets = [set(int(pool.depth[i]) for i in sel)
                      for sel, _ in log if sel]
        self.assertTrue(any(len(ds) > 1 for ds in depth_sets),
                        f"frontier가 depth 혼합 선택을 못함: {depth_sets}")
        # 같은 설정의 level은 항상 단일 depth (== f)
        pool2, log2 = self._run("level", R=6, W=3, F=3)
        for f, (sel, _) in enumerate(log2):
            for i in sel:
                self.assertEqual(int(pool2.depth[i]), f)

    def test_d11_single_shot(self):
        pool, log = self._run("frontier")
        seen = set()
        for sel, _ in log:
            for i in sel:
                self.assertNotIn(i, seen)   # 재평가 없음
                seen.add(i)
                self.assertEqual(int(pool.state[i]), 1)

    def test_budget_conserved(self):
        from ssd.engine.helpers.p2_tree import alloc_root_budgets
        pool, log = self._run("frontier")
        piv = torch.tensor([0.5 / (1.6 ** r) for r in range(4)])  # _run과 동일
        budgets = alloc_root_budgets(piv, total=12, beta=0.5, cap=6)
        # root별 생성된 자식 수 ≤ 배정 예산
        for r in range(4):
            kids = sum(1 for i in range(pool.n)
                       if int(pool.root[i]) == r and int(pool.parent_idx[i]) >= 0)
            self.assertLessEqual(kids, int(budgets[r]))

    def test_priority_is_logpiv_plus_lograwq(self):
        pool, _ = self._run("level")
        import math
        for i in range(pool.n):
            p = int(pool.parent_idx[i])
            if p < 0:
                continue
            expect = float(pool.logpri[p]) + math.log(max(float(pool.raw_q[i]), 1e-9))
            self.assertAlmostEqual(float(pool.logpri[i]), expect, places=5)

    def test_ancestor_cells_chain(self):
        pool, _ = self._run("level")
        # depth 2 노드의 조상 셀 수 == 2 (부모, 조부모 — root는 cell 보유 시)
        for i in range(pool.n):
            if int(pool.depth[i]) == 2 and int(pool.state[i]) == 1:
                cells = pool.ancestors_cells(i)
                self.assertEqual(len(cells), 2)
                break

    def test_cell_addressing(self):
        pool, log = self._run("level")
        for f, (sel, _) in enumerate(log):
            for k, i in enumerate(sel):
                self.assertEqual(int(pool.cell[i]), f * 4 + k)


class TestTreeMask(unittest.TestCase):
    def _chain_reference(self, s, MQ, Kg, ctx, glue_rows):
        # cudagraph_helpers의 chain 빌더 재현 (대각 spec 블록)
        import numpy as np
        cols = ctx + s * MQ
        prefix = cols - ((s + 1) * MQ + (Kg + 1))
        m = np.zeros((MQ, cols), dtype=np.uint8)
        m[:, :prefix] = 1
        m[:, prefix:prefix + Kg + 1] = glue_rows
        d0 = prefix + Kg + 1
        rows = np.arange(MQ)
        for blk in range(s + 1):
            m[rows, d0 + blk * MQ + rows] = 1
        return np.packbits(m.ravel(), bitorder="little")

    def test_chain_degenerate_bit_identical(self):
        # fanout=1 퇴화: 행 k의 조상 = 자기 열의 이전 블록들 → 체인
        # 대각과 비트 단위 일치해야 한다 (fast-path mask 게이트).
        import numpy as np
        from ssd.engine.helpers.p2_tree import build_tree_mask_packed
        MQ, Kg, ctx = 10, 4, 700
        glue = np.tile(np.tril(np.ones((Kg + 1, Kg + 1), np.uint8))[0:1],
                       (MQ, 1))
        glue = np.tril(np.ones((Kg + 1, Kg + 1), np.uint8))
        glue_rows = np.repeat(glue, [2, 2, 2, 2, 2], axis=0)  # fan합=10
        for s in range(3):
            anc = [[b * MQ + k for b in range(s)] for k in range(MQ)]
            selfc = [s * MQ + k for k in range(MQ)]
            packed, _ = build_tree_mask_packed(
                s, MQ, Kg, ctx, glue_rows, anc, selfc)
            ref = self._chain_reference(s, MQ, Kg, ctx, glue_rows)
            self.assertTrue((packed == ref).all(), f"step {s} mismatch")

    def test_tree_cross_row_bits(self):
        # 트리: 행 1의 부모가 (fwd0, 행 0) 셀 → 그 비트가 켜져야 함
        import numpy as np
        from ssd.engine.helpers.p2_tree import build_tree_mask_packed
        MQ, Kg, ctx = 4, 2, 100
        glue_rows = np.ones((MQ, Kg + 1), np.uint8)
        anc = [[], [0], [], []]                # 행1 ← 셀0 (행0@fwd0)
        selfc = [MQ + k for k in range(MQ)]    # fwd=1
        packed, _ = build_tree_mask_packed(
            1, MQ, Kg, ctx, glue_rows, anc, selfc)
        m = np.unpackbits(packed, bitorder="little")
        cols = ctx + MQ
        m = m[:MQ * cols].reshape(MQ, cols)
        spec0 = cols - ((1 + 1) * MQ + (Kg + 1)) + Kg + 1
        self.assertEqual(m[1, spec0 + 0], 1)   # 조상 셀
        self.assertEqual(m[0, spec0 + 0], 0)   # 행0은 셀0 안 봄 (자기 과거 아님)
        self.assertEqual(m[1, spec0 + MQ + 1], 1)  # 자기 셀


class TestRunRollout(unittest.TestCase):
    def test_topology_matches_reference(self):
        # 같은 RNG로 run_rollout(stub forward)과 rollout_reference의
        # topology 장부가 완전히 일치해야 한다.
        import numpy as np
        from ssd.engine.helpers.p2_tree import (rollout_reference,
                                                run_rollout,
                                                tree_sample_wor)
        R, W, F, C, V = 4, 4, 3, 3, 64
        piv = torch.tensor([0.5 / (1.6 ** r) for r in range(R)])
        temps = torch.full((W,), 0.7)
        g = torch.Generator().manual_seed(42)
        fixed_logits = torch.randn(F, W, V, generator=g)

        def fwd(f, ids, rope, packed, indptr):
            return fixed_logits[f].clone()

        torch.manual_seed(7)
        pool_a, log_a, _cl = run_rollout(
            list(range(10, 10 + R)), piv, policy="level", W=W, F_total=F,
            c_tensor=C, nv=6, beta=0.5, depth_cap=4, temps=temps,
            forward_fn=fwd, glue_rows_by_root=np.ones((R, 5), np.uint8),
            rope_base_by_root=[100 + r for r in range(R)], K_glue=4,
            context_len=700)

        torch.manual_seed(7)
        def sample_fn(sel, fan):
            f = sample_fn.calls
            sample_fn.calls += 1
            toks, raws = tree_sample_wor(fixed_logits[f].clone(), temps,
                                         C)
            return toks[:len(sel)], raws[:len(sel)]
        sample_fn.calls = 0
        pool_b, log_b = rollout_reference(
            list(range(10, 10 + R)), piv, None, policy="level", W=W,
            F_total=F, c_tensor=C, nv=6, beta=0.5, depth_cap=4,
            sample_fn=sample_fn)

        self.assertEqual(pool_a.n, pool_b.n)
        for fld in ("tok", "parent_idx", "depth", "root", "sib_order",
                    "cell", "state"):
            ta = getattr(pool_a, fld)[:pool_a.n].tolist()
            tb = getattr(pool_b, fld)[:pool_b.n].tolist()
            self.assertEqual(ta, tb, f"{fld} 불일치")

    def test_rope_uses_depth_not_forward(self):
        import numpy as np
        from ssd.engine.helpers.p2_tree import run_rollout
        seen = {}
        def fwd(f, ids, rope, packed, indptr):
            seen[f] = rope.clone()
            return torch.randn(4, 32)
        torch.manual_seed(1)
        _ = run_rollout([10], torch.tensor([1.0]), policy="frontier", W=4,
                    F_total=3, c_tensor=3, nv=8, beta=0.5, depth_cap=4,
                    temps=torch.full((4,), 0.7), forward_fn=fwd,
                    glue_rows_by_root=np.ones((1, 5), np.uint8),
                    rope_base_by_root=[500], K_glue=4, context_len=600)
        self.assertEqual(int(seen[0][0]), 500)      # root: depth 0
        # 이후 forward의 유효 행 rope는 500+depth (depth는 1 이상)
        self.assertGreaterEqual(int(seen[1][0]), 501)


class TestRootViews(unittest.TestCase):
    def test_views_invariants(self):
        from ssd.engine.helpers.p2_tree import (build_root_views,
                                                rollout_reference)
        import torch as T
        R = 4
        piv = T.tensor([0.5 / (1.6 ** r) for r in range(R)])
        g = T.Generator().manual_seed(3)
        def sample_fn(sel, fan):
            n = len(sel)
            return (T.randint(100, 200, (n, 3), generator=g),
                    T.rand(n, 3, generator=g) * 0.3)
        T.manual_seed(7)
        pool, _ = rollout_reference(
            list(range(10, 10 + R)), piv, None, policy="frontier", W=4,
            F_total=3, c_tensor=3, nv=6, beta=0.5, depth_cap=4,
            sample_fn=sample_fn)
        v = build_root_views(pool, R, nv=6)
        # 유효 수 = pool의 자식 수와 일치, nv 이하
        for r in range(R):
            kids = sum(1 for i in range(pool.n)
                       if int(pool.root[i]) == r and int(pool.parent_idx[i]) >= 0)
            self.assertEqual(int(v["valid"][r]), kids)
            self.assertLessEqual(kids, 6)
        # parent_local < 자기 인덱스 (보행 invariant), pad는 -1/0
        for r in range(R):
            n = int(v["valid"][r])
            for j in range(n):
                self.assertLess(int(v["parent_local"][r, j]), j)
            for j in range(n, 6):
                self.assertEqual(int(v["tok"][r, j]), 0)


class TestSelectorPivPassthrough(unittest.TestCase):
    def test_piv_follows_token_indexing(self):
        # 관통된 piv가 토큰과 동일한 dedup/take/정렬 경로를 따라야 함
        from ssd.engine.draft_runner import DraftRunner
        B, N, P, FO = 1, 12, 5, 2
        chosen_pos = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1]])
        chosen_tok = torch.arange(100, 112).view(1, N)
        piv = torch.linspace(0.5, 0.05, N).view(1, N)
        draft_forked = torch.full((B, P, FO), -1, dtype=torch.int64)
        draft_forked[0, 0, 0] = 100          # rank0 후보는 P1 중복 → dedup
        out = DraftRunner._select_proxy_sourced_tokens_unified(
            {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok,
             "chosen_piv": piv},
            draft_forked, K_rank=4, total_budget=8)
        result, fo_t, taken_piv = out
        # dedup된 100의 piv(0.5)는 빠지고, 각 슬롯 piv == 원 piv[tok-100]
        for j in range(8):
            t = int(result[0, j])
            self.assertNotEqual(t, 100)
            self.assertAlmostEqual(float(taken_piv[0, j]),
                                   float(piv[0, t - 100]), places=6)

    def test_no_piv_keeps_two_tuple(self):
        from ssd.engine.draft_runner import DraftRunner
        chosen_pos = torch.zeros(1, 4, dtype=torch.int64)
        chosen_tok = torch.arange(4).view(1, 4) + 10
        draft_forked = torch.full((1, 5, 2), -1, dtype=torch.int64)
        out = DraftRunner._select_proxy_sourced_tokens_unified(
            {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok},
            draft_forked, K_rank=4, total_budget=4)
        self.assertEqual(len(out), 2)


class TestParentQRefs(unittest.TestCase):
    def test_parent_q_matches_cell_logits(self):
        import numpy as np
        from ssd.engine.helpers.p2_tree import build_root_views, run_rollout
        R, W, F, C, V = 3, 3, 3, 2, 32
        piv = torch.tensor([0.5, 0.3, 0.2])
        fixed = torch.randn(F, W, V)
        def fwd(f, ids, rope, packed, indptr):
            return fixed[f].clone()
        torch.manual_seed(11)
        pool, log, cell_logits = run_rollout(
            [7, 8, 9], piv, policy="level", W=W, F_total=F, c_tensor=C,
            nv=6, beta=0.5, depth_cap=4, temps=torch.full((W,), 0.7),
            forward_fn=fwd, glue_rows_by_root=np.ones((R, 5), np.uint8),
            rope_base_by_root=[10, 20, 30], K_glue=4, context_len=100)
        v = build_root_views(pool, R, nv=6, cell_logits=cell_logits)
        # 각 노드의 parent_q_logits[ref] == cell_logits[부모 셀]
        cnt = [0] * R
        for i in range(pool.n):
            if int(pool.parent_idx[i]) < 0:
                continue
            r = int(pool.root[i]); j = cnt[r]; cnt[r] += 1
            ref = int(v["parent_q_ref"][r, j])
            self.assertGreaterEqual(ref, 0)
            pc = int(pool.parent_cell[i])
            self.assertTrue(torch.equal(v["parent_q_logits"][r, ref],
                                        cell_logits[pc]))
        # u_valid ≤ valid ≤ nv
        for r in range(R):
            self.assertLessEqual(int(v["u_valid"][r]), int(v["valid"][r]) or 1)


class TestLosslessExhaustive(unittest.TestCase):
    """v6 §10 go/no-go ① — 작은 vocab 전수 분포-일치 (하드 게이트)."""

    @staticmethod
    def _ladder(p, q, order):
        # 형제 그룹 사다리 해석해: 주어진 WOR 순서에 대해
        # (j별 도달·수락 확률, 전원기각 잔차 분포)
        def norm(d):
            Z = sum(d.values())
            return {k: v / Z for k, v in d.items()} if Z > 1e-12 else {}
        R = dict(p); D = dict(q)
        reach = 1.0; acc = []
        for tok in order:
            a = min(1.0, R.get(tok, 0.0) / D[tok]) if D.get(tok, 0) > 0 else 0.0
            acc.append((tok, reach * a))
            reach *= (1.0 - a)
            R = norm({k: max(0.0, R.get(k, 0.0) - D.get(k, 0.0))
                      for k in set(R) | set(D)})
            D = norm({k: v for k, v in D.items() if k != tok})
        return acc, reach, R

    @staticmethod
    def _wor_orders(q, f):
        # 비복원 순서 전수 열거 (chain rule 확률)
        import itertools
        toks = [t for t in q if q[t] > 0]
        out = []
        for perm in itertools.permutations(toks, min(f, len(toks))):
            pr = 1.0; rem = 1.0
            for t in perm:
                pr *= q[t] / rem
                rem -= q[t]
            out.append((list(perm), pr))
        return out

    def _exact_first_token_dist(self, p, q, f):
        # Σ_순서 P(순서) × [수락 j → tok_j, 전원기각 → 잔차]
        from collections import defaultdict
        out = defaultdict(float)
        for order, pr in self._wor_orders(q, f):
            acc, reach, R = self._ladder(p, q, order)
            for tok, a in acc:
                out[tok] += pr * a
            for k, v in (R if R else p).items():
                out[k] += pr * reach * v
        return dict(out)

    def test_exhaustive_first_token_equals_p(self):
        import random
        rnd = random.Random(0)
        for trial in range(30):
            V = 4
            pw = [rnd.random() + 1e-3 for _ in range(V)]
            qw = [rnd.random() + 1e-3 for _ in range(V)]
            if trial % 5 == 0:
                qw = list(pw)                      # p=q 케이스
            if trial % 7 == 0:
                pw[0] = 1e-9                       # 반-희소 케이스
            p = {i: w / sum(pw) for i, w in enumerate(pw)}
            q = {i: w / sum(qw) for i, w in enumerate(qw)}
            for f in (1, 2, 3):
                out = self._exact_first_token_dist(p, q, f)
                for k in p:
                    self.assertAlmostEqual(out.get(k, 0.0), p[k], places=9,
                        msg=f"trial{trial} f={f} tok{k}")

    def test_walk_mc_matches_analytic(self):
        # tree_verify_walk 구현이 해석해와 일치하는지 (단일 그룹, MC)
        import random
        from ssd.engine.helpers.p2_tree import tree_verify_walk
        p = {0: 0.5, 1: 0.3, 2: 0.2}
        q = {0: 0.6, 1: 0.3, 2: 0.1}
        f = 2
        rnd = random.Random(42)
        N = 200_000
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(N):
            # WOR 샘플 (chain rule)
            toks = [0, 1, 2]; w = dict(q); order = []
            for _j in range(f):
                z = sum(w.values()); r = rnd.random() * z; accum = 0.0
                for t in list(w):
                    accum += w[t]
                    if r <= accum:
                        order.append(t); del w[t]; break
            view = {"valid": torch.tensor(f),
                    "tok": torch.tensor(order),
                    "parent_local": torch.tensor([-1] * f),
                    "sib_order": torch.tensor(list(range(f)))}
            path, term = tree_verify_walk(
                view, {j: p for j in range(f)}, {-1: q}, p, rnd)
            first = order[path[0]] if path else term
            counts[first] += 1
        for k in p:
            self.assertAlmostEqual(counts[k] / N, p[k], delta=0.005,
                                   msg=f"tok{k}: {counts[k]/N} vs {p[k]}")


class TestTreeWire(unittest.TestCase):
    def test_roundtrip(self):
        import numpy as np
        from ssd.engine.helpers.p2_tree import (build_root_views,
                                                pack_tree_ints,
                                                parse_tree_ints,
                                                run_rollout,
                                                tree_wire_ints_len)
        R, W, F, V, NV = 3, 3, 3, 32, 6
        piv = torch.tensor([0.5, 0.3, 0.2])
        fixed = torch.randn(F, W, V)
        torch.manual_seed(4)
        pool, _, cl = run_rollout(
            [7, 8, 9], piv, policy="level", W=W, F_total=F, c_tensor=2,
            nv=NV, beta=0.5, depth_cap=4, temps=torch.full((W,), 0.7),
            forward_fn=lambda f, *a: fixed[f].clone(),
            glue_rows_by_root=np.ones((R, 5), np.uint8),
            rope_base_by_root=[10, 20, 30], K_glue=4, context_len=100)
        v = build_root_views(pool, R, nv=NV, cell_logits=cl)
        for r in range(R):
            buf = pack_tree_ints(v, r, NV)
            self.assertEqual(buf.numel(), tree_wire_ints_len(NV))
            got = parse_tree_ints(buf, NV)
            self.assertEqual(got["valid"], int(v["valid"][r]))
            self.assertEqual(got["tok"].tolist(), v["tok"][r].tolist())
            self.assertEqual(got["parent_local"].tolist(),
                             v["parent_local"][r].tolist())
        # miss: 전부 0, 크기 동일 (max-padded)
        z = pack_tree_ints(v, -1, NV)
        self.assertEqual(int(z.abs().sum()), 0)
        self.assertEqual(z.numel(), tree_wire_ints_len(NV))


class TestVerifyRows(unittest.TestCase):
    def test_rows_and_mask(self):
        from ssd.engine.helpers.p2_tree import (build_verify_mask_packed,
                                                build_verify_rows)
        import numpy as np
        # 트리: n0(root직결) ← n1, n0 ← n2, n1 ← n3
        ti = {"valid": 4,
              "tok": torch.tensor([11, 12, 13, 14]),
              "parent_local": torch.tensor([-1, 0, 0, 1])}
        bt = torch.arange(10, dtype=torch.int64) + 100   # block ids
        out = build_verify_rows(ti, nv=8, pos0=50, block_table=bt,
                                block_size=16)
        self.assertEqual(out["depth"].tolist(), [0, 1, 1, 2])
        self.assertEqual(out["rope"].tolist(), [51, 52, 52, 53])
        # scratch: pos 51..54 → block 100+3=103 (51//16=3), offset 51%16
        self.assertEqual(int(out["slot"][0]), 103 * 16 + 3)
        self.assertEqual(out["ancestors"][3], [0, 1])
        packed = build_verify_mask_packed(4, out["ancestors"], kv_len=55)
        m = np.unpackbits(packed.numpy(), bitorder="little")[:4 * 55]
        m = m.reshape(4, 55)
        pre = 55 - 4
        self.assertTrue((m[:, :pre] == 1).all())          # 프리픽스
        self.assertEqual(m[3, pre + 0], 1)                # 조상 n0
        self.assertEqual(m[3, pre + 1], 1)                # 조상 n1
        self.assertEqual(m[3, pre + 2], 0)                # 형제 아님-조상
        self.assertEqual(m[3, pre + 3], 1)                # 자기


class TestTensorWalkEquivalence(unittest.TestCase):
    def test_same_coins_same_outcome(self):
        # 같은 분포·같은 코인열에서 dict-참조와 텐서 보행이 동일 결과
        import random
        from ssd.engine.helpers.p2_tree import (tree_verify_walk,
                                                tree_verify_walk_tensor)
        V = 8
        rnd = random.Random(0)
        for trial in range(50):
            # 트리: 2 자식 + 첫 자식 밑 1 자식
            ti = {"valid": 3,
                  "tok": torch.tensor([rnd.randrange(V) for _ in range(3)]),
                  "parent_local": torch.tensor([-1, -1, 0]),
                  "sib_order": torch.tensor([0, 1, 0]),
                  "parent_q_ref": torch.tensor([0, 0, 1])}
            if int(ti["tok"][0]) == int(ti["tok"][1]):
                continue                       # 형제 중복 배제 (비복원)
            pl = torch.randn(4, V)
            qp = torch.softmax(torch.randn(2, V), dim=-1)
            coins = [rnd.random() for _ in range(8)]
            # 참조 dict 세계 구성
            p_d = {c: {k: float(torch.softmax(pl[c + 1 if c >= 0 else 0]
                                              .float() / 0.7, -1)[k])
                       for k in range(V)} for c in (-1, 0, 1, 2)}
            q_d = {-1: {k: float(qp[0][k]) for k in range(V)},
                   0: {k: float(qp[1][k]) for k in range(V)},
                   1: {k: float(qp[1][k]) for k in range(V)},
                   2: {k: float(qp[1][k]) for k in range(V)}}
            class R1:
                def __init__(self): self.i = 0
                def random(self):
                    v = coins[self.i]; self.i += 1; return v
            r1, r2 = R1(), R1()
            view = {"valid": torch.tensor(3), "tok": ti["tok"],
                    "parent_local": ti["parent_local"],
                    "sib_order": ti["sib_order"]}
            path_a, term_a = tree_verify_walk(
                view, {j: p_d[j] for j in range(3)}, q_d, p_d[-1], r1)
            # 텐서판: mult_fn을 같은 코인으로 CDF 샘플
            def mult(probs):
                r = r2.random(); acc = 0.0
                for k in range(V):
                    acc += float(probs[k])
                    if r <= acc:
                        return k
                return V - 1
            path_b, term_b = tree_verify_walk_tensor(
                ti, pl, qp, 0.7, coin_fn=r2.random, mult_fn=mult)
            self.assertEqual(path_a, path_b, f"trial {trial}")
            self.assertEqual(term_a, term_b, f"trial {trial}")


class TestCommitPlan(unittest.TestCase):
    def test_plan_and_identity_skip(self):
        from ssd.engine.helpers.p2_tree import commit_copy_plan
        bt = torch.arange(8, dtype=torch.int64) + 50
        # 경로 [0, 2]: 노드0 → dst 자리 그대로 (skip), 노드2 → dst k=1
        plan = commit_copy_plan([0, 2], pos0=30, block_table=bt,
                                block_size=16)
        self.assertEqual(len(plan), 1)
        src, dst = plan[0]
        self.assertEqual(src, int(bt[33 // 16]) * 16 + 33 % 16)
        self.assertEqual(dst, int(bt[32 // 16]) * 16 + 32 % 16)

class TestBudgetExhaustion(unittest.TestCase):
    """이슈 #23 (리뷰 2A): 예산 완전 소진 — sum == min(total, R·cap)."""

    def test_skewed_piv_full_exhaustion(self):
        import ssd.engine.helpers.p2_tree as PT
        torch.manual_seed(0)
        for beta in (0.5, 1.0, 2.0):
            for cap in (6, 8):
                for _ in range(50):
                    piv = torch.distributions.Dirichlet(
                        torch.full((10,), 0.3)).sample()
                    b = PT.alloc_root_budgets(piv, total=40, beta=beta,
                                              cap=cap)
                    self.assertEqual(int(b.sum()),
                                     min(40, 10 * cap),
                                     f"beta={beta} cap={cap}")
                    self.assertTrue(bool((b <= cap).all()))

    def test_zero_piv_excluded(self):
        # 이슈 #24: piv==0 root는 water-filling 포화 후에도 예산 0
        import ssd.engine.helpers.p2_tree as PT
        piv = torch.tensor([0.5, 0.3, 0.0, 0.0])
        b = PT.alloc_root_budgets(piv, total=40, beta=0.5, cap=8)
        self.assertEqual(int(b[2]), 0)
        self.assertEqual(int(b[3]), 0)
        self.assertEqual(int(b.sum()), 16)   # 유자격 2×cap

    def test_cap_binds_total(self):
        import ssd.engine.helpers.p2_tree as PT
        piv = torch.ones(3)
        b = PT.alloc_root_budgets(piv, total=40, beta=0.5, cap=8)
        self.assertEqual(int(b.sum()), 24)   # R·cap = 24 < 40


class TestBackboneFanout(unittest.TestCase):
    """형상 진단(docs/duet/21 §4.5) 수정: backbone-우선 배분의 형상 보장."""

    def _run(self, budgets_override, c_tensor, depth_cap=4, W=10, F=4,
             policy="level"):
        import ssd.engine.helpers.p2_tree as PT
        R = len(budgets_override)
        toks = list(range(100, 100 + R))
        piv = torch.ones(R)
        orig = PT.alloc_root_budgets
        try:
            PT.alloc_root_budgets = lambda *a, **k: torch.tensor(
                budgets_override, dtype=torch.int64)
            def sample_fn(sel, fan):
                n = len(sel)
                C = max(1, int(fan.max())) if hasattr(fan, "max") else 1
                t = torch.arange(n * C).view(n, C) + 1000
                q = torch.full((n, C), 0.5)
                return t, q
            pool, _ = PT.rollout_reference(
                toks, piv, None, policy=policy, W=W, F_total=F,
                c_tensor=c_tensor, nv=8, beta=0.5, depth_cap=depth_cap,
                sample_fn=sample_fn, fanout_policy="backbone")
        finally:
            PT.alloc_root_budgets = orig
        # root별 최대 깊이/노드 수
        dmax = [0] * R
        cnt = [0] * R
        for i in range(pool.n):
            if int(pool.parent_idx[i]) < 0:
                continue
            r = int(pool.root[i])
            dmax[r] = max(dmax[r], int(pool.depth[i]))
            cnt[r] += 1
        return dmax, cnt

    def test_budget8_full_depth_plus_siblings(self):
        # 예산 8 → 백본 4 + 형제 4 (E1 승리 형상)
        dmax, cnt = self._run([8], c_tensor=3)
        self.assertEqual(dmax[0], 4)
        self.assertEqual(cnt[0], 8)

    def test_budget4_pure_chain(self):
        # 예산 4 = depth_cap → 순수 체인 (형제 0, C와 무관)
        dmax, cnt = self._run([4], c_tensor=3)
        self.assertEqual(dmax[0], 4)
        self.assertEqual(cnt[0], 4)

    def test_budget2_partial_backbone(self):
        # 예산 2 → 깊이 2 부분 백본 (최대한 깊게)
        dmax, cnt = self._run([2], c_tensor=3)
        self.assertEqual(dmax[0], 2)
        self.assertEqual(cnt[0], 2)

    def test_multi_root_each_backboned(self):
        # 여러 root 혼합 예산 — 각자 min(budget, depth_cap) 깊이 보장
        dmax, cnt = self._run([8, 4, 2, 0], c_tensor=3)
        self.assertEqual(dmax, [4, 4, 2, 0])
        self.assertEqual(cnt, [8, 4, 2, 0])

    def test_ctensor_policy_unchanged(self):
        # 기존 정책은 그대로 (회귀 가드): budget 4, C=3 → 폭 소진 깊이 2
        import ssd.engine.helpers.p2_tree as PT
        orig = PT.alloc_root_budgets
        try:
            PT.alloc_root_budgets = lambda *a, **k: torch.tensor(
                [4], dtype=torch.int64)
            def sample_fn(sel, fan):
                n = len(sel)
                C = max(1, int(fan.max()))
                return torch.arange(n * C).view(n, C) + 1000, \
                    torch.full((n, C), 0.5)
            pool, _ = PT.rollout_reference(
                [100], torch.ones(1), None, policy="level", W=10,
                F_total=4, c_tensor=3, nv=8, beta=0.5, depth_cap=4,
                sample_fn=sample_fn, fanout_policy="ctensor")
        finally:
            PT.alloc_root_budgets = orig
        dmax = max((int(pool.depth[i]) for i in range(pool.n)
                    if int(pool.parent_idx[i]) >= 0), default=0)
        self.assertLessEqual(dmax, 2)

