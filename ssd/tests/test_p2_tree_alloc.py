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
