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
