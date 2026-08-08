"""T1.2 unit tests — P2-tree 사전 예산 배분 (docs/duet/internal/20).

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_p2_tree_alloc
"""
import unittest
from unittest.mock import patch

import torch

from ssd.engine.helpers.p2_tree import (
    alloc_fanouts, alloc_fanouts_global, alloc_policy_root_budgets,
    alloc_root_budgets)
from ssd.config import Config
import ssd.engine.helpers.p2_tree as PT


class TestMultiwordAncestry(unittest.TestCase):
    def test_mask_pack_crosses_63_bit_boundaries(self):
        max_cells = 145
        ar = PT.TreeArena(4, "cpu", max_cells=max_cells)
        self.assertEqual(ar.anc_words, 3)
        marked = (0, 62, 63, 64, 125, 126, 144)
        for cell in marked:
            word, bit = divmod(cell, PT._ANC_WORD_BITS)
            ar.anc_bits[0, word] |= 1 << bit

        # f=145,W=1 gives 145 previous cells plus the current self cell.
        packed, _ = PT._arena_mask_pack(
            145, 1, 0, 2, torch.ones(1, 1, dtype=torch.uint8),
            ar.anc_bits[:1], torch.ones(1, dtype=torch.bool), "cpu")
        bits = ((packed.unsqueeze(1) >> torch.arange(8)) & 1) \
            .to(torch.uint8).reshape(-1)[:147]
        # col0=glue, col1..145=previous cells, col146=current self.
        self.assertEqual(int(bits[0]), 1)
        for cell in range(145):
            self.assertEqual(int(bits[1 + cell]), int(cell in marked))
        self.assertEqual(int(bits[146]), 1)


class TestP2ExecutorWarmup(unittest.TestCase):
    def test_capture_does_not_consume_production_rng(self):
        """Startup capture may execute sampling, but must be RNG-neutral."""
        import os
        from types import SimpleNamespace
        from ssd.engine.draft_runner import DraftRunner

        gen = torch.Generator().manual_seed(1234)
        before = gen.get_state().clone()
        executor = SimpleNamespace(F=3, W=10, gen=gen, graphs={})

        def capture(bucket):
            # Model the graph warmup's real sampling side effect.
            torch.rand(4, generator=gen)
            executor.graphs[bucket] = object()

        executor.capture = capture
        executor.prime_capture_inputs = lambda bucket: None
        runner = SimpleNamespace(
            config=SimpleNamespace(
                duet_tree_policy="confidence", use_eagle=False,
                max_model_len=2048, max_blocks=8),
            block_size=256, device=torch.device("cpu"),
            _ensure_p2_exec=lambda: executor)

        env = {"SSD_TREE_EXEC": "1", "SSD_TREE_EXEC_WARMUP": "all"}
        with patch.dict(os.environ, env, clear=False), \
                patch("torch.cuda.synchronize"), \
                patch("torch.cuda.mem_get_info", return_value=(8 << 30,
                                                                16 << 30)):
            DraftRunner._warmup_p2_tree_executor(runner)

        self.assertEqual(sorted(executor.graphs), list(range(1, 8)))
        self.assertTrue(torch.equal(before, gen.get_state()))


class TestTreeVerifyMaskPacking(unittest.TestCase):
    def test_flat_little_endian_layout(self):
        mask = torch.tensor([
            [1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ], dtype=torch.bool)
        packed = PT.pack_tree_verify_mask(mask)
        bits = torch.tensor([
            (int(packed[i // 8]) >> (i % 8)) & 1
            for i in range(mask.numel())
        ], dtype=torch.bool)
        self.assertTrue(torch.equal(bits, mask.reshape(-1)))

    def test_requires_two_dimensional_cpu_mask(self):
        with self.assertRaisesRegex(ValueError, "2-D CPU"):
            PT.pack_tree_verify_mask(torch.ones(8, dtype=torch.bool))


class TestTreeWireValidation(unittest.TestCase):
    @staticmethod
    def _tree():
        # Two root siblings share qref 0; node 2 extends node 0 from qref 1.
        return {
            "valid": 3,
            "u_valid": 2,
            "epoch": 1,
            "tok": torch.tensor([7, 8, 9, 0]),
            "parent_local": torch.tensor([-1, -1, 0, 0]),
            "sib_order": torch.tensor([0, 1, 0, 0]),
            "parent_q_ref": torch.tensor([0, 0, 1, 0]),
        }

    def test_valid_topology(self):
        tree = self._tree()
        self.assertIs(PT.validate_tree_ints(tree, 4, vocab_size=16), tree)

    def test_parse_and_validate_accepts_shm_list_without_tensor_copy(self):
        nv = 4
        buf = [3, 2, 1,
               7, 8, 9, 0,
               -1, -1, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0]
        tree = PT.parse_tree_ints(buf, nv)
        self.assertIs(PT.validate_tree_ints(tree, nv, vocab_size=16), tree)
        self.assertEqual(tree["parent_local"][:3], [-1, -1, 0])

    def test_self_parent_is_rejected(self):
        tree = self._tree()
        tree["parent_local"][2] = 2
        with self.assertRaisesRegex(RuntimeError, "parent invariant"):
            PT.validate_tree_ints(tree, 4)

    def test_forward_parent_is_rejected(self):
        tree = self._tree()
        tree["parent_local"][0] = 2
        with self.assertRaisesRegex(RuntimeError, "parent invariant"):
            PT.validate_tree_ints(tree, 4)

    def test_invalid_qref_is_rejected(self):
        tree = self._tree()
        tree["parent_q_ref"][2] = 2
        with self.assertRaisesRegex(RuntimeError, "parent-q ref"):
            PT.validate_tree_ints(tree, 4)

    def test_sibling_gap_is_rejected(self):
        tree = self._tree()
        tree["sib_order"][1] = 3
        with self.assertRaisesRegex(RuntimeError, "sibling order"):
            PT.validate_tree_ints(tree, 4)

    def test_siblings_must_share_qref(self):
        tree = self._tree()
        tree["parent_q_ref"][1] = 1
        with self.assertRaisesRegex(RuntimeError, "multiple q refs"):
            PT.validate_tree_ints(tree, 4)

    def test_token_range_is_rejected(self):
        tree = self._tree()
        tree["tok"][1] = 16
        with self.assertRaisesRegex(RuntimeError, "token out of range"):
            PT.validate_tree_ints(tree, 4, vocab_size=16)

    def test_parent_path_is_bounded_and_root_ordered(self):
        tree = self._tree()
        self.assertEqual(PT.tree_parent_path(tree, 0), [])
        self.assertEqual(PT.tree_parent_path(tree, 1), [0])
        self.assertEqual(PT.tree_parent_path(tree, 3), [0, 2])
        with self.assertRaisesRegex(RuntimeError, "outside"):
            PT.tree_parent_path(tree, 5)

    def test_parent_path_rejects_cycle_even_without_wire_validation(self):
        tree = self._tree()
        tree["parent_local"][2] = 2
        with self.assertRaisesRegex(RuntimeError, "strictly decreasing"):
            PT.tree_parent_path(tree, 3)


class TestTreeP1Allocation(unittest.TestCase):
    def test_chain_backbone_floor_is_preserved(self):
        # backbone 0->2->4->5 plus root siblings 1,3
        par = [-1, -1, 0, 0, 2, 4]
        sib = [0, 1, 0, 1, 0, 0]
        raw = [0.8, 0.2, 0.7, 0.3, 0.6, 0.5]
        counts = PT.allocate_tree_p1_fanouts(
            par, sib, raw, total_budget=16,
            chain_fanouts=[2, 2, 2, 2, 2])
        # contexts: root=0; backbone nodes 0,2,4,5 -> 1,3,5,6
        for ctx in (0, 1, 3, 5, 6):
            self.assertGreaterEqual(counts[ctx], 2)
        self.assertEqual(sum(counts), 16)
        # Both non-backbone terminal contexts retain cache coverage.
        self.assertGreaterEqual(counts[2], 1)
        self.assertGreaterEqual(counts[4], 1)

    def test_confidence_breaks_surplus_ties(self):
        counts = PT.allocate_tree_p1_fanouts(
            [-1, -1], [0, 1], [0.9, 0.1], total_budget=6,
            chain_fanouts=[1, 1])
        self.assertEqual(sum(counts), 6)
        self.assertGreater(counts[1], counts[2])

    def test_invalid_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            PT.allocate_tree_p1_fanouts(
                [-1], [], [0.5], 2, [1, 1])


class TestRootBudgets(unittest.TestCase):
    def test_global_root_count_defaults_to_width_and_honors_arg(self):
        cfg = object.__new__(Config)
        cfg.duet_tree_policy = "hybrid"
        cfg.duet_tree_root_count = None
        cfg.duet_p2_budget = 10
        cfg.duet_phase2_k = 4
        cfg.speculate_k = 13
        self.assertEqual(cfg.duet_p2_active_root_count, 10)
        self.assertEqual(cfg.duet_p2_seed_count, 10)
        cfg.duet_tree_root_count = 7
        self.assertEqual(cfg.duet_p2_active_root_count, 7)
        self.assertEqual(cfg.duet_p2_seed_count, 7)

    def test_canonical_confidence_root_count_is_derived(self):
        cfg = object.__new__(Config)
        cfg.duet_tree_policy = "confidence"
        cfg.duet_p2_budget = 10
        cfg.duet_phase2_k = 4
        cfg.speculate_k = 13
        # 4 rounds x width 10, with 4 backbone + 2 rescue nodes/root.
        self.assertEqual(cfg.duet_p2_active_root_count, 6)
        self.assertEqual(cfg.duet_p2_seed_count, 6)
        with patch.dict("os.environ", {"SSD_TREE_ROOT_SHADOW": "1"}):
            self.assertEqual(cfg.duet_p2_active_root_count, 6)
            self.assertEqual(cfg.duet_p2_seed_count, 10)
        # Legacy policies keep their explicit reproduction knob.
        cfg.duet_tree_policy = "level"
        cfg.duet_tree_root_count = 7
        self.assertEqual(cfg.duet_p2_seed_count, 7)

    def test_coverage_policy_keeps_every_live_root(self):
        cfg = object.__new__(Config)
        cfg.duet_tree_policy = "coverage"
        # Even a stale legacy override cannot silently violate the property;
        # Config validation rejects it in a fully initialized Config.
        cfg.duet_tree_root_count = 3
        cfg.duet_p2_budget = 10
        cfg.duet_phase2_k = 4
        cfg.speculate_k = 13
        self.assertEqual(cfg.duet_p2_active_root_count, 10)
        self.assertEqual(cfg.duet_p2_seed_count, 10)

        piv = torch.tensor([0.7, 0.2, 0.1, 0.0])
        budgets = alloc_policy_root_budgets(
            piv, "coverage", total=16, beta=0.5, cap=8)
        # Stored child count is intentionally larger than the 16 parent
        # forward cells.  The final zero is a padding/non-live root.
        self.assertEqual(budgets.tolist(), [8, 8, 8, 0])
        self.assertGreater(int(budgets.sum()), 16)

    def test_coverage_tree_is_chain_superset_for_all_ten_roots(self):
        calls = 0

        def sample_fn(sel, fan):
            nonlocal calls
            # Distinct first child per (round,lane); siblings are ordered
            # after it and therefore cannot replace the chain backbone.
            rows = len(sel)
            base = 1000 + calls * 100
            toks = (base + torch.arange(rows).unsqueeze(1) * 3
                    + torch.arange(3).unsqueeze(0))
            raw = torch.tensor([[0.6, 0.25, 0.1]]).repeat(rows, 1)
            calls += 1
            return toks, raw

        R = W = 10
        piv = torch.linspace(1.0, 0.1, R)
        pool, _ = PT.rollout_reference(
            list(range(10, 20)), piv, None, policy="coverage", W=W,
            F_total=4, c_tensor=3, nv=8, beta=0.5, depth_cap=4,
            sample_fn=sample_fn, fanout_policy="backbone")
        views = PT.build_root_views(pool, R, 8)
        self.assertEqual(views["valid"].tolist(), [8] * R)

        expected_par = [-1, -1, -1, 0, 0, 0, 3, 6]
        expected_sib = [0, 1, 2, 0, 1, 2, 0, 0]
        for r in range(R):
            self.assertEqual(views["parent_local"][r].tolist(),
                             expected_par)
            self.assertEqual(views["sib_order"][r].tolist(), expected_sib)
            # The canonical first-child chain has four nodes and is retained
            # unchanged; every other node is an added sibling leaf.
            chain = []
            parent = -1
            for _ in range(4):
                child = next(j for j, (p, s) in enumerate(zip(
                    expected_par, expected_sib)) if p == parent and s == 0)
                chain.append(child)
                parent = child
            self.assertEqual(chain, [0, 3, 6, 7])

            q = views["raw_q"][r]
            prefix = []
            for j, p in enumerate(expected_par):
                prefix.append(float(q[j]) * (1.0 if p < 0 else prefix[p]))
            chain_mass = sum(prefix[j] for j in chain)
            tree_mass = sum(prefix)
            self.assertGreater(tree_mass, chain_mass)

    def test_eagle_expands_global_confidence_not_every_root_backbone(self):
        calls = 0

        def sample_fn(sel, fan):
            nonlocal calls
            rows = len(sel)
            toks = (1000 + calls * 100
                    + torch.arange(rows).unsqueeze(1) * 3
                    + torch.arange(3).unsqueeze(0))
            # The high root's first children dominate the low root even when
            # local q is equal, because the score begins with root P_iv.
            raw = torch.tensor([[0.80, 0.15, 0.04]]).repeat(rows, 1)
            calls += 1
            return toks, raw

        pool, eval_log = PT.rollout_reference(
            [10, 11], torch.tensor([0.95, 0.05]), None,
            policy="eagle", W=2, F_total=4, c_tensor=3, nv=8,
            beta=0.5, depth_cap=4, sample_fn=sample_fn,
            fanout_policy="backbone")  # eagle overrides this legacy knob
        # Round zero evaluates both roots.  Later rounds need not spend a
        # mandatory lane on the weak root.
        self.assertEqual(
            sorted(int(pool.root[i]) for i in eval_log[0][0]), [0, 1])
        self.assertTrue(any(
            eval_log[f][0]
            and all(int(pool.root[i]) == 0 for i in eval_log[f][0])
            for f in range(1, 4)))
        views = PT.build_root_views(pool, 2, 8)
        self.assertGreater(int(views["valid"][0]),
                           int(views["valid"][1]))
        self.assertGreaterEqual(int(views["valid"][1]), 1)

    def test_dynamic_is_the_same_global_selector_as_legacy_eagle(self):
        def run(policy):
            calls = 0

            def sample_fn(sel, fan):
                nonlocal calls
                rows = len(sel)
                toks = (3000 + calls * 100
                        + torch.arange(rows).unsqueeze(1) * 3
                        + torch.arange(3).unsqueeze(0))
                raw = torch.tensor([[0.80, 0.15, 0.04]]).repeat(rows, 1)
                calls += 1
                return toks, raw

            pool, trace = PT.rollout_reference(
                [10, 11], torch.tensor([0.95, 0.05]), None,
                policy=policy, W=2, F_total=4, c_tensor=3, nv=8,
                beta=0.5, depth_cap=4, sample_fn=sample_fn,
                fanout_policy="backbone")
            return pool, trace

        old, old_trace = run("eagle")
        new, new_trace = run("dynamic")
        self.assertEqual(old.n, new.n)
        for field in ("tok", "parent_idx", "depth", "root", "sib_order",
                      "state", "cell"):
            self.assertTrue(torch.equal(
                getattr(old, field)[:old.n], getattr(new, field)[:new.n]),
                field)
        for (old_sel, old_fan), (new_sel, new_fan) in zip(
                old_trace, new_trace):
            self.assertEqual(old_sel, new_sel)
            self.assertTrue(torch.equal(old_fan, new_fan))

    def test_hybrid_guarantees_depth_two_then_expands_globally(self):
        calls = 0

        def sample_fn(sel, fan):
            nonlocal calls
            rows = len(sel)
            toks = (2000 + calls * 100
                    + torch.arange(rows).unsqueeze(1) * 3
                    + torch.arange(3).unsqueeze(0))
            raw = torch.tensor([[0.80, 0.15, 0.04]]).repeat(rows, 1)
            calls += 1
            return toks, raw

        pool, eval_log = PT.rollout_reference(
            [10, 11], torch.tensor([0.95, 0.05]), None,
            policy="hybrid", W=2, F_total=4, c_tensor=3, nv=8,
            beta=0.5, depth_cap=4, sample_fn=sample_fn,
            fanout_policy="backbone")
        # Both roots own a mandatory tip in rounds zero and one.  Only after
        # reaching depth two may the stronger root consume all forward lanes.
        for f in (0, 1):
            self.assertEqual(
                sorted(int(pool.root[i]) for i in eval_log[f][0]), [0, 1])
        self.assertTrue(any(
            eval_log[f][0]
            and all(int(pool.root[i]) == 0 for i in eval_log[f][0])
            for f in (2, 3)))
        views = PT.build_root_views(pool, 2, 8)
        for r in range(2):
            n = int(views["valid"][r])
            depth = [0] * n
            for j in range(n):
                p = int(views["parent_local"][r, j])
                depth[j] = 1 if p < 0 else depth[p] + 1
            self.assertGreaterEqual(max(depth), 2)

    def test_eagle_rejects_more_roots_than_forward_width(self):
        with self.assertRaisesRegex(ValueError, "R<=W"):
            PT.rollout_reference(
                [10, 11, 12], torch.tensor([0.6, 0.3, 0.1]), None,
                policy="eagle", W=2, F_total=2, c_tensor=2, nv=4,
                beta=0.5, depth_cap=2,
                sample_fn=lambda sel, fan: (
                    torch.ones(len(sel), 2, dtype=torch.int64),
                    torch.full((len(sel), 2), 0.4)),
                fanout_policy="backbone")

    def test_calibrated_floors_only_block_later_expansion(self):
        # Root 0 is below the proxy floor.  Node 3 is below the local-q
        # floor.  Both roots must still run in round zero and both children
        # remain valid leaves; only node 4 may receive a later forward.
        pool = PT.TreePool(8)
        pool.add(10, -1, -1, 0, 0, 0, float(torch.log(
            torch.tensor(0.002))), 1.0)
        pool.add(11, -1, -1, 0, 1, 0, float(torch.log(
            torch.tensor(0.1))), 1.0)
        pool.add(20, 0, 0, 1, 0, 0, -7.0, 0.9)    # low proxy root
        pool.add(21, 1, 1, 1, 1, 0, -7.0, 0.005)  # low confidence
        pool.add(22, 1, 1, 1, 1, 1, -2.0, 0.2)    # eligible
        remaining = torch.tensor([4, 4])

        roots = PT.select_nodes_global(
            pool, 4, 0, 4, remaining, future_rounds=3,
            proxy_threshold=0.003, conf_threshold=0.01)
        self.assertEqual(sorted(roots), [0, 1])
        pool.state[:2] = 1
        chosen = PT.select_nodes_global(
            pool, 4, 1, 4, remaining, future_rounds=2,
            proxy_threshold=0.003, conf_threshold=0.01)
        self.assertEqual(chosen, [4])
        self.assertEqual(pool.n, 5)  # floors did not delete either leaf

        ar = PT.TreeArena(8, "cpu")
        n = pool.n
        for name, src in (("tok", pool.tok),
                          ("parent_idx", pool.parent_idx),
                          ("parent_cell", pool.parent_cell),
                          ("depth", pool.depth), ("root", pool.root),
                          ("logpri", pool.logpri), ("raw_q", pool.raw_q),
                          ("state", pool.state), ("cell", pool.cell)):
            getattr(ar, name)[:n].copy_(src[:n])
        ar.valid[:n] = True
        ar.n.fill_(n)
        sel, valid = PT._arena_select_global(
            ar, 4, 1, 4, remaining, future_rounds=2, R=2,
            proxy_threshold=0.003, conf_threshold=0.01)
        self.assertEqual(int(valid.sum()), 1)
        self.assertEqual(int(sel[0]), 4)

    def test_adaptive_preserves_every_backbone_and_only_varies_siblings(self):
        calls = 0

        def sample_fn(sel, fan):
            nonlocal calls
            rows = len(sel)
            toks = (2000 + calls * 100
                    + torch.arange(rows).unsqueeze(1) * 3
                    + torch.arange(3).unsqueeze(0))
            raw = torch.tensor([[0.80, 0.15, 0.04]]).repeat(rows, 1)
            calls += 1
            return toks, raw

        pool, _ = PT.rollout_reference(
            [10, 11], torch.tensor([0.90, 0.10]), None,
            policy="adaptive", W=2, F_total=4, c_tensor=3, nv=8,
            beta=0.5, depth_cap=4, sample_fn=sample_fn,
            fanout_policy="backbone")
        views = PT.build_root_views(pool, 2, 8)
        # Both roots retain a four-token first-child chain.  Only the strong
        # root receives optional siblings and therefore a wider view.
        self.assertGreater(int(views["valid"][0]),
                           int(views["valid"][1]))
        self.assertEqual(int(views["valid"][1]), 4)
        for r in range(2):
            par = views["parent_local"][r]
            sib = views["sib_order"][r]
            cur = -1
            depth = 0
            while True:
                kids = [j for j in range(int(views["valid"][r]))
                        if int(par[j]) == cur and int(sib[j]) == 0]
                if not kids:
                    break
                cur = kids[0]
                depth += 1
            self.assertEqual(depth, 4)

    def test_sum_and_cap(self):
        piv = torch.tensor([0.5, 0.2, 0.1, 0.05])
        b = alloc_root_budgets(piv, total=40, beta=0.5, cap=8)
        self.assertLessEqual(int(b.sum()), 40)
        self.assertTrue(bool((b <= 8).all()))
        self.assertTrue(bool((b >= 0).all()))
        # cap이 배분을 제한하는 레짐: 상위 root가 cap에 닿는다
        self.assertEqual(int(b[0]), 8)

    def test_confidence_policy_fixes_beta_and_matches_level(self):
        def sample_fn(sel, fan):
            rows = len(sel)
            toks = torch.arange(rows * 3).view(rows, 3) + 100
            raw = torch.tensor([[0.7, 0.2, 0.1]]).repeat(rows, 1)
            return toks, raw

        piv = torch.tensor([0.55, 0.25, 0.12, 0.08])
        a, _ = PT.rollout_reference(
            [10, 11, 12, 13], piv, None, policy="confidence", W=4,
            F_total=3, c_tensor=3, nv=6, beta=1.0, depth_cap=3,
            sample_fn=sample_fn, fanout_policy="backbone")
        b, _ = PT.rollout_reference(
            [10, 11, 12, 13], piv, None, policy="level", W=4,
            F_total=3, c_tensor=3, nv=6, beta=0.5, depth_cap=3,
            sample_fn=sample_fn, fanout_policy="backbone")
        for field in ("tok", "parent_idx", "root", "depth", "sib_order"):
            self.assertEqual(
                getattr(a, field)[:a.n].tolist(),
                getattr(b, field)[:b.n].tolist())

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
    def test_global_fanout_reserves_depth_only_for_selected_roots(self):
        pri = torch.tensor([3.0, 2.0, 1.0])
        root = torch.tensor([0, 0, 0])
        # Five slots remain with two future rounds.  Three may be used now;
        # breadth-first allocation gives one child to every selected parent.
        f = alloc_fanouts_global(
            pri, root, torch.tensor([5]), c_tensor=3, future_rounds=2)
        self.assertEqual(f.tolist(), [1, 1, 1])
        # One selected parent receives the same three-token fanout.
        f1 = alloc_fanouts_global(
            pri[:1], root[:1], torch.tensor([5]), c_tensor=3,
            future_rounds=2)
        self.assertEqual(f1.tolist(), [3])

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

    def test_out_of_range_wire_positions_are_excluded_without_oob(self):
        from ssd.engine.draft_runner import DraftRunner
        chosen_pos = torch.tensor([[99, -3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
        chosen_tok = torch.arange(12).view(1, 12) + 100
        piv = torch.linspace(1.0, 0.1, 12).view(1, 12)
        draft_forked = torch.full((1, 5, 2), -1, dtype=torch.int64)
        result, fan, taken_piv = \
            DraftRunner._select_proxy_sourced_tokens_unified(
                {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok,
                 "chosen_piv": piv},
                draft_forked, K_rank=4, total_budget=8)
        self.assertEqual(int(fan.sum()), 8)
        self.assertTrue(bool((result >= 102).all()))
        self.assertTrue(bool((taken_piv < 1.0).all()))


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
            # 1b: lazy pq — cells 참조로 검증 (물질화는 서빙 시 gather)
            self.assertEqual(int(v["parent_q_cells"][r, ref]), pc)
            self.assertTrue(torch.equal(
                v["cell_logits"][int(v["parent_q_cells"][r, ref])],
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

        # A smaller phase-local view can ride in a larger shared P1/P2 wire.
        wire_nv = 13
        buf = pack_tree_ints(v, 0, wire_nv)
        got = parse_tree_ints(buf, wire_nv)
        self.assertEqual(got["valid"], int(v["valid"][0]))
        self.assertEqual(got["tok"][:NV].tolist(), v["tok"][0].tolist())
        self.assertEqual(got["tok"][NV:].tolist(), [0] * (wire_nv - NV))
        self.assertEqual(got["parent_local"][NV:].tolist(),
                         [0] * (wire_nv - NV))


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

class TestTerminalMassDP(unittest.TestCase):
    """이슈 #25: 트리 Policy-B 종단질량 DP."""

    def test_chain_degenerate_equals_first_reject(self):
        import ssd.engine.helpers.p2_tree as PT
        a = torch.tensor([0.9, 0.8, 0.7, 0.6])
        term = PT.terminal_mass_dp([-1, 0, 1, 2], a)
        chain = [0.1, 0.9 * 0.2, 0.9 * 0.8 * 0.3,
                 0.9 * 0.8 * 0.7 * 0.4, 0.9 * 0.8 * 0.7 * 0.6]
        for i, c in enumerate(chain):
            self.assertAlmostEqual(float(term[i]), c, places=6)
        self.assertAlmostEqual(float(term.sum()), 1.0, places=6)

    def test_sibling_mass_and_total(self):
        import ssd.engine.helpers.p2_tree as PT
        # root 자식 둘 (형제): [-1,-1], 뒤형제는 앞형제 기각 조건
        a = torch.tensor([0.5, 0.5])
        term = PT.terminal_mass_dp([-1, -1], a)
        # rec 종단 = (1-.5)(1-.5)=.25; 노드0 종단 = .5(잎); 노드1 = .25
        self.assertAlmostEqual(float(term[0]), 0.25, places=6)
        self.assertAlmostEqual(float(term[1]), 0.5, places=6)
        self.assertAlmostEqual(float(term[2]), 0.25, places=6)
        self.assertAlmostEqual(float(term.sum()), 1.0, places=6)


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
    """형상 진단(docs/duet/internal/21 §4.5) 수정: backbone-우선 배분의 형상 보장."""

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



class TestReview2Fixes(unittest.TestCase):
    """리뷰2 수용 (이슈 #27/#28/#33) — docs/duet/internal/20."""

    # --- #33: 비례 water-filling ---
    def test_proportional_after_cap(self):
        # 리뷰2-7 예제: cap-절단 질량은 잔여 가중치 '비례'로 재배분
        # (종전 라운드-로빈은 [8,5,3]로 균등화)
        b = PT.alloc_root_budgets(torch.tensor([0.9, 0.09, 0.01]),
                                  total=16, beta=1.0, cap=8)
        self.assertEqual(b.tolist(), [8, 7, 1])

    # --- #27: W-경합에서 tip 의무 lane ---
    def test_tip_lane_under_w_contention(self):
        # 재현 픽스처: 예산 [7,7,7,7,6,6], W=10, F=4, C=3 — 종전
        # top-W 컷이 약root tip을 탈락시켜 34/40 생성, dmax=1 정지.
        forced = torch.tensor([7, 7, 7, 7, 6, 6], dtype=torch.int64)
        orig = PT.alloc_root_budgets
        try:
            PT.alloc_root_budgets = lambda *a, **k: forced.clone()
            g = torch.Generator().manual_seed(11)

            def sample_fn(sel, fan):
                n = len(sel)
                C = max(1, int(fan.max())) if len(fan) else 1
                return (torch.randint(100, 2000, (n, C), generator=g),
                        torch.rand(n, C, generator=g) * 0.5)

            piv = torch.tensor([0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
            pool, _ = PT.rollout_reference(
                list(range(10, 16)), piv, None, policy="level", W=10,
                F_total=4, c_tensor=3, nv=7, beta=0.5, depth_cap=4,
                sample_fn=sample_fn, fanout_policy="backbone")
        finally:
            PT.alloc_root_budgets = orig
        st = pool.alloc_stats
        self.assertEqual(st["generated"], st["allocated"],
                         f"예산 소실: {st}")
        # 모든 root의 backbone이 depth_cap=4까지 연장 (예산 6-7 ≥ 4)
        self.assertTrue(all(d == 4 for d in st["per_root_dmax"]),
                        f"약root 깊이 정지: {st['per_root_dmax']}")

    def test_backbone_r_gt_w_raises(self):
        with self.assertRaises(ValueError):
            PT.rollout_reference(
                list(range(10, 22)), torch.rand(12), None,
                policy="level", W=10, F_total=4, c_tensor=3, nv=7,
                beta=0.5, depth_cap=4,
                sample_fn=lambda s, f: (torch.zeros(len(s), 3,
                                                    dtype=torch.int64),
                                        torch.ones(len(s), 3) * 0.3),
                fanout_policy="backbone")

    # --- #28: 정확 sibling ladder ---
    def test_ladder_counterexample_vs_independent(self):
        # 2형제 반례 (수계산): q=[.5,.25,.25], p^E=[.25,.25,.5], toks 0,1
        # 독립 근사: α=(.5, 1) → all-reject 0.
        # 사다리: a1=.5; R←norm((p−q)+)=[0,0,1], D←[0,.5,.5];
        #         a2=min(1, 0/.5)=0 → all-reject (1−.5)(1−0)=0.5.
        p_rows = torch.tensor([[0.25, 0.25, 0.5],
                               [1 / 3, 1 / 3, 1 / 3],
                               [1 / 3, 1 / 3, 1 / 3]])
        q_rows = torch.tensor([[0.5, 0.25, 0.25]] * 2)
        toks = torch.tensor([0, 1])
        alpha, term, resid = PT.tree_policy_b_ladder(
            [-1, -1], [0, 1], toks, p_rows, q_rows)
        self.assertAlmostEqual(float(alpha[0]), 0.5, places=6)
        self.assertAlmostEqual(float(alpha[1]), 0.0, places=6)
        self.assertAlmostEqual(float(term[0]), 0.5, places=6)  # rec 종단
        # 독립 근사는 0 — 정정 확인
        ind = PT.terminal_mass_dp([-1, -1], torch.tensor([0.5, 1.0]))
        self.assertAlmostEqual(float(ind[0]), 0.0, places=6)
        # rec ctx residual = 사다리 최종 R = [0,0,1]
        self.assertTrue(torch.allclose(
            resid[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))

    def test_ladder_chain_degenerate_matches_dp(self):
        # C=1 사슬: 형제 없음 → 조건부 == 독립, DP와 일치해야 함
        torch.manual_seed(5)
        V = 8
        p_rows = torch.softmax(torch.randn(4, V), dim=-1)
        q_rows = torch.softmax(torch.randn(3, V), dim=-1)
        toks = torch.randint(0, V, (3,))
        par = [-1, 0, 1]
        alpha, term, _ = PT.tree_policy_b_ladder(
            par, [0, 0, 0], toks, p_rows, q_rows)
        ind = PT.terminal_mass_dp(par, alpha)
        self.assertTrue(torch.allclose(term, ind, atol=1e-5),
                        f"{term} vs {ind}")
        for j in range(3):
            a_ref = min(1.0, float(p_rows[par[j] + 1, toks[j]])
                        / (float(q_rows[j, toks[j]]) + 1e-10))
            self.assertAlmostEqual(float(alpha[j]), a_ref, places=5)

    def test_ladder_terminal_mass_sums_to_one(self):
        torch.manual_seed(7)
        V = 16
        # rec→(n0,n1), n0→(n2,n3), n1→n4 — 형제 2쌍 포함
        par = [-1, -1, 0, 0, 1]
        sib = [0, 1, 0, 1, 0]
        p_rows = torch.softmax(torch.randn(6, V), dim=-1)
        q_rows = torch.softmax(torch.randn(5, V) * 0.7, dim=-1)
        # 노드별 q row = 부모 컨텍스트 분포 (형제는 동일 row)
        q_rows[1] = q_rows[0]
        q_rows[3] = q_rows[2]
        toks = torch.tensor([2, 5, 1, 9, 4])
        _, term, resid = PT.tree_policy_b_ladder(
            par, sib, toks, p_rows, q_rows)
        self.assertAlmostEqual(float(term.sum()), 1.0, places=5)
        for r in range(6):
            self.assertAlmostEqual(float(resid[r].sum()), 1.0, places=4)

    def test_ladder_matches_walk_monte_carlo(self):
        # 골드 테스트: p_logits=log p^E로 보행을 돌리면 (proxy=truth)
        # 보행의 종단 ctx 빈도 ≈ ladder term (같은 수학의 두 구현).
        torch.manual_seed(13)
        V = 6
        par = [-1, -1, 0, 0, 1]
        sib = [0, 1, 0, 1, 0]
        toks = torch.tensor([0, 1, 2, 3, 4])
        p_rows = torch.softmax(torch.randn(6, V), dim=-1)
        q_base = torch.softmax(torch.randn(3, V) * 0.5, dim=-1)
        # q_rows[j] = 부모 ctx의 제안분포; parent_q_ref: rec→0, n0→1, n1→2
        q_rows = q_base[torch.tensor([0, 0, 1, 1, 2])]
        alpha, term, resid = PT.tree_policy_b_ladder(
            par, sib, toks, p_rows, q_rows)
        ints = {"valid": 5, "tok": toks.tolist(), "parent_local": par,
                "sib_order": sib,
                "parent_q_ref": [0, 0, 1, 1, 2]}
        g = torch.Generator().manual_seed(99)
        counts = torch.zeros(6)
        N = 30000
        p_logits = torch.log(p_rows.clamp_min(1e-30))
        for _ in range(N):
            def coin():
                return float(torch.rand(1, generator=g))

            def mult(pr):
                return int(torch.multinomial(pr, 1, generator=g))
            path, _t = PT.tree_verify_walk_tensor(
                ints, p_logits, q_base, 1.0, coin, mult)
            counts[(1 + path[-1]) if path else 0] += 1
        freq = counts / N
        for c in range(6):
            self.assertAlmostEqual(float(freq[c]), float(term[c]),
                                   delta=0.012,
                                   msg=f"ctx {c}: walk {freq.tolist()} "
                                       f"vs ladder {term.tolist()}")

    def test_ladder_gpu_parity(self):
        if not torch.cuda.is_available():
            self.skipTest("no cuda")
        torch.manual_seed(3)
        V = 12
        par = [-1, -1, 0, 0, 1]
        sib = [0, 1, 0, 1, 0]
        toks = torch.randint(0, V, (5,))
        p_rows = torch.softmax(torch.randn(6, V), dim=-1)
        q_rows = torch.softmax(torch.randn(5, V), dim=-1)
        a_c, t_c, r_c = PT.tree_policy_b_ladder(par, sib, toks, p_rows,
                                                q_rows)
        a_g, t_g, r_g = PT.tree_policy_b_ladder(
            par, sib, toks.cuda(), p_rows.cuda(), q_rows.cuda())
        self.assertTrue(torch.allclose(a_c, a_g.cpu(), atol=1e-5))
        self.assertTrue(torch.allclose(t_c, t_g.cpu(), atol=1e-5))
        self.assertTrue(torch.allclose(r_c, r_g.cpu(), atol=1e-5))

    def test_fixed_proxy_candidates_are_exact_at_real_width(self):
        # Exact-width capture is required: padding a global top-k can swap
        # bit-identical ties even when all probability values are equal.
        par = [-1, -1, -1, 0, 0, 1, 3, 6]
        sib = [0, 1, 2, 0, 1, 0, 0, 0]
        nv, V, wire = len(par), 257, 24
        for seed in range(10):
            torch.manual_seed(seed)
            exit_logits = torch.randn(nv + 1, V, dtype=torch.bfloat16)
            q_logits = torch.randn(nv, V, dtype=torch.bfloat16)
            toks = torch.randint(V, (nv,))
            p = torch.softmax(exit_logits.float(), -1)
            q = torch.softmax(q_logits.float(), -1)
            _, term, resid = PT.tree_policy_b_ladder(
                par, sib, toks, p, q)
            piv = resid * term.unsqueeze(1)
            piv[torch.tensor(par) + 1, toks] = 0.0
            top_v, top_i = piv.flatten().topk(wire)
            expected_pos = top_i // V
            expected_tok = PT.pack_piv(top_i % V, top_v)
            got_pos, got_tok, _ = PT.tree_proxy_candidates_fixed(
                exit_logits, q_logits, toks,
                PT.pack_tree_proxy_topology(par, sib, nv), wire, 4)
            self.assertTrue(torch.equal(expected_pos, got_pos))
            self.assertTrue(torch.equal(expected_tok, got_tok))

    def test_ladder_dtype_does_not_depend_on_global_default(self):
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            PT.warmup_tree_proxy_kernels(
                torch.zeros(5, 64, dtype=torch.bfloat16),
                torch.zeros(4, 64, dtype=torch.bfloat16),
                [-1, -1, -1, 0], [0, 1, 2, 0], wire_n=12)
        finally:
            torch.set_default_dtype(old)

    def test_proxy_cudagraph_matches_dynamic_all_valid_widths(self):
        if not torch.cuda.is_available():
            self.skipTest("no cuda")
        # Prefixes are all valid breadth/depth-first trees and exercise root
        # siblings, non-root siblings, and a depth-four backbone.
        par8 = [-1, -1, -1, 0, 0, 1, 3, 6]
        sib8 = [0, 1, 2, 0, 1, 0, 0, 0]
        V, wire = 257, 24
        for nv in range(1, 9):
            graph = PT.TreeProxyCUDAGraph(
                nv, V, wire, depth_steps=4, dtype=torch.bfloat16,
                device="cuda")
            par, sib = par8[:nv], sib8[:nv]
            graph.prepare_topology(par, sib)
            for seed in range(3):
                gen = torch.Generator(device="cuda").manual_seed(
                    100 * nv + seed)
                exit_logits = torch.randn(
                    nv + 1, V, dtype=torch.bfloat16, device="cuda",
                    generator=gen)
                q_logits = torch.randn(
                    nv, V, dtype=torch.bfloat16, device="cuda",
                    generator=gen)
                tokens = torch.randint(
                    V, (nv,), dtype=torch.int64, device="cuda",
                    generator=gen)
                expected = PT.tree_proxy_candidates_fixed(
                    exit_logits, q_logits, tokens,
                    PT.pack_tree_proxy_topology(
                        par, sib, nv, device="cuda"),
                    wire, 4)
                got = graph.replay(exit_logits, q_logits, tokens)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(expected[0], got[0]),
                                f"pos nv={nv} seed={seed}")
                self.assertTrue(torch.equal(expected[1], got[1]),
                                f"tok nv={nv} seed={seed}")

    def test_chain_proxy_fixed_matches_policy_b_formula(self):
        K, V, top_k, wire = 4, 257, 6, 24
        for seed in range(8):
            torch.manual_seed(seed)
            exit_logits = torch.randn(K + 1, V, dtype=torch.bfloat16)
            q_logits = torch.randn(K, V, dtype=torch.bfloat16)
            tokens = torch.randint(V, (K,))

            p_e = torch.softmax(exit_logits[:K].float(), -1)
            p_d = torch.softmax(q_logits.float(), -1)
            idx = tokens[:, None]
            accept = (p_e.gather(1, idx).squeeze(1)
                      / (p_d.gather(1, idx).squeeze(1) + 1e-10)).clamp(max=1)
            residual = (p_e - p_d).clamp(min=0)
            residual.scatter_(1, idx, 0.0)
            probs, ids = residual.topk(top_k, -1)
            probs /= probs.sum(-1, keepdim=True).clamp(min=1e-10)
            cp = torch.cumprod(accept, 0)
            h = torch.zeros(K + 1)
            h[0] = 1 - accept[0]
            h[1:K] = cp[:-1] * (1 - accept[1:])
            h[K] = cp[-1]
            p_last = torch.softmax(exit_logits[K].float(), -1)
            last_p, last_id = p_last.topk(top_k)
            last_p /= last_p.sum().clamp(min=1e-10)
            probs = torch.cat([probs, last_p[None]], 0)
            ids = torch.cat([ids, last_id[None]], 0)
            piv = h[:, None] * probs
            top_v, top_i = piv.flatten().topk(wire)
            expected_pos = top_i // top_k
            expected_tok = PT.pack_piv(
                ids.flatten().gather(0, top_i), top_v)

            got_pos, got_tok, got_v = PT.chain_proxy_candidates_fixed(
                exit_logits, q_logits, tokens, top_k, wire, True)
            self.assertTrue(torch.equal(expected_pos, got_pos))
            self.assertTrue(torch.equal(expected_tok, got_tok))
            self.assertTrue(torch.equal(top_v, got_v))

    def test_chain_shaped_tree_proxy_matches_chain_policy(self):
        """C=1 tree must preserve the established chain root ranking."""
        K, V, top_k, wire = 4, 257, 14, 24
        topology = PT.pack_tree_proxy_topology(
            [-1, 0, 1, 2], [0, 0, 0, 0], K)
        for seed in range(8):
            torch.manual_seed(seed)
            exit_logits = torch.randn(K + 1, V, dtype=torch.bfloat16)
            q_logits = torch.randn(K, V, dtype=torch.bfloat16)
            tokens = torch.randint(V, (K,))
            chain = PT.chain_proxy_candidates_fixed(
                exit_logits, q_logits, tokens, top_k, wire, True)
            tree = PT.tree_proxy_candidates_fixed(
                exit_logits, q_logits, tokens, topology, wire, K, top_k)
            self.assertTrue(torch.equal(chain[0], tree[0]), seed)
            self.assertTrue(torch.equal(chain[1], tree[1]), seed)
            self.assertTrue(torch.allclose(
                chain[2], tree[2], atol=2e-7, rtol=2e-6), seed)

    def test_chain_proxy_cudagraph_matches_fixed(self):
        if not torch.cuda.is_available():
            self.skipTest("no cuda")
        K, V, top_k, wire = 4, 257, 6, 24
        graph = PT.ChainProxyCUDAGraph(
            K, V, top_k, wire, True, torch.bfloat16, "cuda")
        for seed in range(4):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            exit_logits = torch.randn(
                K + 1, V, dtype=torch.bfloat16, device="cuda",
                generator=gen)
            q_logits = torch.randn(
                K, V, dtype=torch.bfloat16, device="cuda", generator=gen)
            tokens = torch.randint(
                V, (K,), dtype=torch.int64, device="cuda", generator=gen)
            expected = PT.chain_proxy_candidates_fixed(
                exit_logits, q_logits, tokens, top_k, wire, True)
            got = graph.replay(exit_logits, q_logits, tokens)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(expected[0], got[0]))
            self.assertTrue(torch.equal(expected[1], got[1]))


class TestArenaParity(unittest.TestCase):
    """T6 1a 동등성 게이트 (22번 v2 §3): 같은 입력·시드에서 arena와
    CPU rollout의 라운드 트레이스가 완전 일치해야 한다."""

    def _mk(self, V=64, seed=5, F=4, W=10):
        g = torch.Generator().manual_seed(seed)
        per_f = [torch.randn(W, V, generator=g) for _ in range(F)]
        cap_cpu, cap_ar = [], []

        def fwd_cpu(f, ids, rope, packed, indptr):
            cap_cpu.append((ids.clone(), rope.clone(), packed.copy()))
            return per_f[f].clone()

        def fwd_ar(f, ids, rope, packed, indptr):
            cap_ar.append((ids.cpu().clone(), rope.cpu().clone(),
                           packed.cpu().numpy().copy()))
            return per_f[f].clone()

        return fwd_cpu, fwd_ar, cap_cpu, cap_ar

    def test_budgets_gpu_matches_cpu(self):
        g = torch.Generator().manual_seed(3)
        for _ in range(200):
            R = int(torch.randint(1, 11, (1,), generator=g))
            piv = torch.rand(R, generator=g)
            piv[torch.rand(R, generator=g) < 0.3] = 0.0   # sentinel 혼합
            total = int(torch.randint(1, 41, (1,), generator=g))
            cap = int(torch.randint(1, 9, (1,), generator=g))
            beta = float(torch.rand(1, generator=g))
            a = PT.alloc_root_budgets(piv, total, beta, cap)
            b = PT.alloc_root_budgets_gpu(piv, total, beta, cap)
            self.assertEqual(a.tolist(), b.tolist(),
                             f"piv={piv.tolist()} t={total} c={cap} "
                             f"b={beta}")

    def test_rollout_arena_full_parity(self):
        import numpy as _np
        R, W, F, C = 6, 10, 4, 3
        piv = torch.tensor([0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
        toks = list(range(10, 16))
        glue = _np.zeros((R, 5), dtype=_np.uint8)
        for r in range(R):
            glue[r, :(r % 5) + 1] = 1
        rope_base = [100 + 3 * r for r in range(R)]
        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=8,
                  beta=0.5, depth_cap=4, K_glue=4, context_len=200,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.7))
        fwd_cpu, fwd_ar, cap_cpu, cap_ar = self._mk(W=W, F=F)
        torch.manual_seed(77)
        pool, log, cl_cpu = PT.run_rollout(
            toks, piv, forward_fn=fwd_cpu, glue_rows_by_root=glue,
            rope_base_by_root=rope_base, **kw)
        torch.manual_seed(77)
        ar, trace, cl_ar = PT.run_rollout_arena(
            toks, piv.clone(), forward_fn=fwd_ar,
            glue_rows_by_root=glue, rope_base_by_root=rope_base,
            device="cpu", **kw)
        pool2 = ar.to_pool(R)
        # ① 노드 수·필드 전체
        self.assertEqual(pool.n, pool2.n)
        n = pool.n
        for fld, fld2 in (("tok", "tok"), ("parent_idx", "parent_idx"),
                          ("parent_cell", "parent_cell"),
                          ("depth", "depth"), ("root", "root"),
                          ("sib_order", "sib_order"),
                          ("state", "state"), ("cell", "cell")):
            self.assertEqual(getattr(pool, fld)[:n].tolist(),
                             getattr(pool2, fld2)[:n].tolist(), fld)
        self.assertTrue(torch.allclose(pool.raw_q[:n].double(),
                                       pool2.raw_q[:n].double(),
                                       atol=1e-6))
        lp1 = pool.logpri[:n].double()
        lp2 = pool2.logpri[:n].double()
        fin = torch.isfinite(lp1)
        self.assertTrue(torch.equal(fin, torch.isfinite(lp2)))
        self.assertTrue(torch.allclose(lp1[fin], lp2[fin], atol=1e-9))
        # ② 선택·fanout 라운드 트레이스
        sel_t, val_t, fan_t = trace
        for f in range(F):
            sel_cpu, fan_cpu = log[f]
            n_sel = len(sel_cpu)
            self.assertEqual(int(val_t[f].sum()), n_sel, f"f={f}")
            self.assertEqual(sel_t[f][:n_sel].tolist(), sel_cpu, f"f={f}")
            self.assertEqual(fan_t[f][:n_sel].tolist(),
                             fan_cpu.tolist()
                             if torch.is_tensor(fan_cpu) else fan_cpu,
                             f"f={f}")
        # ③ forward 입력·mask 바이트
        self.assertEqual(len(cap_cpu), len(cap_ar))
        for f, ((i1, r1, p1), (i2, r2, p2)) in enumerate(
                zip(cap_cpu, cap_ar)):
            self.assertEqual(i1.tolist(), i2.tolist(), f"ids f={f}")
            self.assertEqual(r1.tolist(), r2.tolist(), f"rope f={f}")
            self.assertTrue((_np.frombuffer(p1.tobytes(), _np.uint8)
                             == p2).all(), f"mask f={f}")
        # ④ cell_logits (같은 stub — 배선 확인)
        self.assertTrue(torch.allclose(cl_cpu, cl_ar, atol=0))

    def test_rollout_arena_parity_fuzz_seeds(self):
        import numpy as _np
        R, W, F, C = 6, 10, 4, 3
        glue = _np.ones((R, 5), dtype=_np.uint8)
        rope_base = [50] * R
        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=7,
                  beta=1.0, depth_cap=4, K_glue=4, context_len=120,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.9))
        for seed in (1, 2, 9, 42):
            piv = torch.rand(R, generator=torch.Generator()
                             .manual_seed(seed))
            fwd_cpu, fwd_ar, _, _ = self._mk(W=W, F=F, seed=seed)
            torch.manual_seed(seed)
            pool, _, _ = PT.run_rollout(
                list(range(20, 20 + R)), piv, forward_fn=fwd_cpu,
                glue_rows_by_root=glue, rope_base_by_root=rope_base,
                **kw)
            torch.manual_seed(seed)
            ar, _, _ = PT.run_rollout_arena(
                list(range(20, 20 + R)), piv.clone(), forward_fn=fwd_ar,
                glue_rows_by_root=glue, rope_base_by_root=rope_base,
                device="cpu", **kw)
            pool2 = ar.to_pool(R)
            self.assertEqual(pool.n, pool2.n, f"seed={seed}")
            n = pool.n
            self.assertEqual(pool.parent_idx[:n].tolist(),
                             pool2.parent_idx[:n].tolist(),
                             f"seed={seed}")
            self.assertEqual(pool.tok[:n].tolist(),
                             pool2.tok[:n].tolist(), f"seed={seed}")


class TestArenaCudaParity(unittest.TestCase):
    """리뷰5: CPU-장치 패리티만으론 불충분 — 실제 CUDA에서 CPU rollout
    과의 동등성을 CI로 고정 (실형상 R_phys=10/R_active=6, zero-q 포함)."""

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    def test_cuda_parity_live_shape(self):
        import numpy as _np
        R, W, F, C = 10, 10, 4, 3           # R_phys=10 (live)
        piv = torch.tensor([0.4, 0.2, 0.1, 0.06, 0.03, 0.01,
                            0.0, 0.0, 0.0, 0.0])   # R_active=6 (#24)
        toks = list(range(30, 30 + R))
        glue = _np.ones((R, 5), dtype=_np.uint8)
        rope_base = [77] * R
        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=8,
                  beta=0.5, depth_cap=4, K_glue=4, context_len=180,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.7))
        V = 48
        g = torch.Generator().manual_seed(11)
        per_f = [torch.randn(W, V, generator=g) for _ in range(F)]

        def fwd_cpu(f, ids, rope, packed, indptr):
            return per_f[f].clone()

        def fwd_gpu(f, ids, rope, packed, indptr):
            return per_f[f].clone().cuda()

        # RNG: CPU rollout은 CPU 제너레이터, arena(CUDA)는 CUDA 제너레이터
        # — 다른 스트림이므로 '같은 샘플'을 강제하기 위해 raw_q/tok을
        # forward logits이 아니라 결정적 샘플러로 비교하는 대신, 여기선
        # 동일-분포 정책 검증이 목적이 아니라 **위상 규칙 동등성**이
        # 목적 — 같은 토큰이 뽑히도록 온도를 낮춰 결정성 확보.
        kw["temps"] = torch.full((W,), 1e-3)   # 사실상 greedy WOR
        torch.manual_seed(5)
        pool, _, _ = PT.run_rollout(
            toks, piv, forward_fn=fwd_cpu, glue_rows_by_root=glue,
            rope_base_by_root=rope_base, **kw)
        torch.manual_seed(5)
        torch.cuda.manual_seed_all(5)
        ar, _, _ = PT.run_rollout_arena(
            toks, piv.cuda(), forward_fn=fwd_gpu,
            glue_rows_by_root=glue, rope_base_by_root=rope_base,
            device="cuda:0", **kw)
        pool2 = ar.to_pool(R)
        self.assertEqual(pool.n, pool2.n)
        n = pool.n
        self.assertEqual(pool.tok[:n].tolist(), pool2.tok[:n].tolist())
        self.assertEqual(pool.parent_idx[:n].tolist(),
                         pool2.parent_idx[:n].tolist())
        self.assertEqual(pool.root[:n].tolist(),
                         pool2.root[:n].tolist())
        self.assertEqual(pool.cell[:n].tolist(),
                         pool2.cell[:n].tolist())
        # R_active 분리: 무예산 root(6-9)는 자식 0
        gen_by_root = [0] * R
        for i in range(n):
            if int(pool2.parent_idx[i]) >= 0:
                gen_by_root[int(pool2.root[i])] += 1
        self.assertTrue(all(gen_by_root[r] == 0 for r in range(6, 10)),
                        gen_by_root)

    def test_zero_q_support_exhaustion_cpu(self):
        # WOR support < C: zero-q 자식은 양쪽 모두 제외 (#38) — 위상
        # 재매김 후에도 동일 pool이어야 한다.
        import numpy as _np
        R, W, F, C = 2, 4, 2, 3
        piv = torch.tensor([0.7, 0.3])
        glue = _np.ones((R, 3), dtype=_np.uint8)
        V = 3                                # support 3 — C=3과 경계
        g = torch.Generator().manual_seed(2)
        # 한 행은 사실상 one-hot (support 1) → 2·3번째 샘플 zero-q
        base = torch.full((W, V), -30.0)
        base[:, 0] = 5.0
        per_f = [base.clone(), torch.randn(W, V, generator=g)]

        def fwd(f, ids, rope, packed, indptr):
            return per_f[f].clone()

        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=4,
                  beta=0.5, depth_cap=2, K_glue=2, context_len=60,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.7))
        torch.manual_seed(9)
        pool, _, _ = PT.run_rollout(
            [5, 6], piv, forward_fn=fwd, glue_rows_by_root=glue,
            rope_base_by_root=[10, 20], **kw)
        torch.manual_seed(9)
        ar, _, _ = PT.run_rollout_arena(
            [5, 6], piv.clone(), forward_fn=fwd, glue_rows_by_root=glue,
            rope_base_by_root=[10, 20], device="cpu", **kw)
        pool2 = ar.to_pool(R)
        self.assertEqual(pool.n, pool2.n)
        n = pool.n
        self.assertEqual(pool.parent_idx[:n].tolist(),
                         pool2.parent_idx[:n].tolist())
        self.assertEqual(pool.tok[:n].tolist(), pool2.tok[:n].tolist())
        self.assertEqual(pool.sib_order[:n].tolist(),
                         pool2.sib_order[:n].tolist())


class TestArenaParityHardening(unittest.TestCase):
    """리뷰6 강화: near-tie 정렬, 실제 zero-q, 결정적 샘플러 주입으로
    temp 0.7 분기 상태의 전필드 CUDA 패리티."""

    def test_near_tie_mandatory_ordering(self):
        # 반례 회귀: mand 두 tip의 priority가 f32 해상도(@1000)보다
        # 가깝게 붙어도 CPU와 같은 순서로 선택돼야 한다.
        import numpy as _np
        R, W, F, C = 2, 3, 2, 2
        piv = torch.tensor([float(_np.exp(-1.00001)),
                            float(_np.exp(-1.0))])   # logpri -1.00001/-1.0
        glue = _np.ones((R, 3), dtype=_np.uint8)
        V = 8
        g = torch.Generator().manual_seed(4)
        per_f = [torch.randn(W, V, generator=g) for _ in range(F)]

        def fwd(f, ids, rope, packed, indptr):
            return per_f[f].clone()

        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=4,
                  beta=0.5, depth_cap=2, K_glue=2, context_len=60,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.7))
        torch.manual_seed(3)
        pool, log, _ = PT.run_rollout(
            [5, 6], piv, forward_fn=fwd, glue_rows_by_root=glue,
            rope_base_by_root=[10, 20], **kw)
        torch.manual_seed(3)
        ar, trace, _ = PT.run_rollout_arena(
            [5, 6], piv.clone(), forward_fn=fwd, glue_rows_by_root=glue,
            rope_base_by_root=[10, 20], device="cpu", **kw)
        sel_t, val_t, _ = trace
        for f in range(F):
            sel_cpu, _fan = log[f]
            n_sel = len(sel_cpu)
            self.assertEqual(sel_t[f][:n_sel].tolist(), sel_cpu,
                             f"f={f}: near-tie 선택 순서 불일치")

    def test_mandatory_backbones_keep_root_lane_order(self):
        """C=1,R=W must keep root r in lane r on every round."""
        R = W = 4
        F = 3
        # Deliberately reverse score order so score sorting would visibly
        # permute lanes after the first round.
        piv = torch.tensor([0.01, 0.05, 0.2, 0.7])
        glue = torch.ones(R, 2, dtype=torch.uint8)

        def fwd(_f, ids, _rope, _packed, _indptr):
            logits = torch.full((W, 32), -20.0)
            logits[torch.arange(W), (ids + 1) % 32] = 20.0
            return logits

        kwargs = dict(
            policy="backbone", W=W, F_total=F, c_tensor=1, nv=F,
            beta=0.5, depth_cap=F, K_glue=1, context_len=64,
            pad_token=0, fanout_policy="backbone",
            temps=torch.full((W,), 0.8))
        torch.manual_seed(17)
        pool, log, _ = PT.run_rollout(
            [3, 7, 11, 15], piv, forward_fn=fwd,
            glue_rows_by_root=glue, rope_base_by_root=[20] * R, **kwargs)
        torch.manual_seed(17)
        ar, trace, _ = PT.run_rollout_arena(
            [3, 7, 11, 15], piv, forward_fn=fwd,
            glue_rows_by_root=glue, rope_base_by_root=[20] * R,
            device="cpu", **kwargs)

        expected_roots = list(range(R))
        sel_gpu, valid_gpu, _ = trace
        for f in range(F):
            sel_cpu, _ = log[f]
            self.assertEqual(
                [int(pool.root[i]) for i in sel_cpu], expected_roots)
            self.assertTrue(bool(valid_gpu[f].all()))
            self.assertEqual(
                ar.root.gather(0, sel_gpu[f]).tolist(), expected_roots)

    def test_true_zero_q_support_exhaustion(self):
        # -inf logits → 정확히 0 확률 (리뷰6: 종전 -30은 1.9e-22 양수)
        import numpy as _np
        R, W, F, C = 2, 4, 2, 3
        piv = torch.tensor([0.7, 0.3])
        glue = _np.ones((R, 3), dtype=_np.uint8)
        V = 3
        base = torch.full((W, V), float("-inf"))
        base[:, 0] = 1.0                      # support 1 < C=3
        g = torch.Generator().manual_seed(2)
        per_f = [base.clone(), torch.randn(W, V, generator=g)]

        def fwd(f, ids, rope, packed, indptr):
            return per_f[f].clone()

        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=4,
                  beta=0.5, depth_cap=2, K_glue=2, context_len=60,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.7))
        torch.manual_seed(9)
        pool, _, _ = PT.run_rollout(
            [5, 6], piv, forward_fn=fwd, glue_rows_by_root=glue,
            rope_base_by_root=[10, 20], **kw)
        torch.manual_seed(9)
        ar, _, _ = PT.run_rollout_arena(
            [5, 6], piv.clone(), forward_fn=fwd, glue_rows_by_root=glue,
            rope_base_by_root=[10, 20], device="cpu", **kw)
        pool2 = ar.to_pool(R)
        # f=0에서 root당 zero-q 자식 최대 2개 배제 발생 확인
        f0_kids = sum(1 for i in range(pool.n)
                      if int(pool.parent_idx[i]) >= 0
                      and int(pool.depth[i]) == 1)
        self.assertLessEqual(f0_kids, R)      # support 1 → root당 1
        self.assertEqual(pool.n, pool2.n)
        n = pool.n
        self.assertEqual(pool.parent_idx[:n].tolist(),
                         pool2.parent_idx[:n].tolist())
        self.assertEqual(pool.tok[:n].tolist(), pool2.tok[:n].tolist())
        self.assertTrue(torch.allclose(pool.raw_q[:n], pool2.raw_q[:n],
                                       atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    def test_cuda_full_parity_deterministic_sampler(self):
        # RNG 스트림 차이를 제거하기 위해 tree_sample_wor를 결정적
        # 스텁으로 치환 — temp 0.7 실분기 상태에서 전필드+트레이스
        # CUDA 패리티 (리뷰6: 종전 1e-3 greedy는 분기 미검증).
        import numpy as _np
        R, W, F, C = 10, 10, 4, 3
        piv = torch.tensor([0.4, 0.2, 0.1, 0.06, 0.03, 0.01,
                            0.0, 0.0, 0.0, 0.0])
        glue = _np.ones((R, 5), dtype=_np.uint8)
        V = 32
        g = torch.Generator().manual_seed(21)
        per_f = [torch.randn(W, V, generator=g) for _ in range(F)]
        det_state = {"calls": 0}
        orig = PT.tree_sample_wor

        def det_wor(logits, temps, c, sampler_x=None, F=None,
                    assume_pos_temps=False):
            gg = torch.Generator().manual_seed(1000 + det_state["calls"])
            det_state["calls"] += 1
            Wl, Vl = logits.shape
            toks = torch.stack([torch.randperm(Vl, generator=gg)[:c]
                                for _ in range(Wl)]).to(logits.device)
            raws = (torch.rand(Wl, c, generator=gg) * 0.5 + 0.01) \
                .to(logits.device)
            return toks, raws

        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=8,
                  beta=0.5, depth_cap=4, K_glue=4, context_len=180,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.7))
        try:
            PT.tree_sample_wor = det_wor
            det_state["calls"] = 0
            pool, log, _ = PT.run_rollout(
                list(range(30, 40)), piv,
                forward_fn=lambda f, i, r, p, ip: per_f[f].clone(),
                glue_rows_by_root=glue, rope_base_by_root=[7] * R, **kw)
            det_state["calls"] = 0
            cap_masks = []

            def fwd_gpu(f, ids, rope, packed, indptr):
                cap_masks.append(packed.cpu().numpy().copy())
                return per_f[f].clone().cuda()

            ar, trace, _ = PT.run_rollout_arena(
                list(range(30, 40)), piv.cuda(), forward_fn=fwd_gpu,
                glue_rows_by_root=glue, rope_base_by_root=[7] * R,
                device="cuda:0", **kw)
        finally:
            PT.tree_sample_wor = orig
        pool2 = ar.to_pool(R)
        self.assertEqual(pool.n, pool2.n)
        n = pool.n
        for fld in ("tok", "parent_idx", "parent_cell", "depth", "root",
                    "sib_order", "state", "cell"):
            self.assertEqual(getattr(pool, fld)[:n].tolist(),
                             getattr(pool2, fld)[:n].tolist(), fld)
        self.assertTrue(torch.allclose(pool.raw_q[:n], pool2.raw_q[:n],
                                       atol=1e-6))
        lp1, lp2 = pool.logpri[:n], pool2.logpri[:n]
        fin = torch.isfinite(lp1)
        self.assertTrue(torch.equal(fin, torch.isfinite(lp2)))
        self.assertTrue(torch.allclose(lp1[fin], lp2[fin], atol=1e-5))
        # 선택 트레이스도 비교
        sel_t, val_t, fan_t = trace
        for f in range(F):
            sel_cpu, fan_cpu = log[f]
            ns = len(sel_cpu)
            self.assertEqual(sel_t[f][:ns].cpu().tolist(), sel_cpu)
            self.assertEqual(
                fan_t[f][:ns].cpu().tolist(),
                fan_cpu.tolist() if torch.is_tensor(fan_cpu) else fan_cpu)

    def test_workspace_reuse_parity(self):
        # persistent arena: 같은 workspace로 연속 2회 rollout — 2회차가
        # fresh와 동일해야 함 (stale 상태 누출 검사)
        import numpy as _np
        R, W, F, C = 4, 6, 3, 3
        glue = _np.ones((R, 4), dtype=_np.uint8)
        kw = dict(policy="level", W=W, F_total=F, c_tensor=C, nv=6,
                  beta=0.5, depth_cap=3, K_glue=3, context_len=90,
                  pad_token=0, fanout_policy="backbone",
                  temps=torch.full((W,), 0.8))
        V = 16
        g = torch.Generator().manual_seed(6)
        per_f = [torch.randn(W, V, generator=g) for _ in range(F)]

        def fwd(f, ids, rope, packed, indptr):
            return per_f[f].clone()

        ws = {}
        piv1 = torch.tensor([0.5, 0.3, 0.1, 0.05])
        piv2 = torch.tensor([0.05, 0.4, 0.4, 0.1])
        # 1회차 (큰 트리로 workspace 오염 유도)
        torch.manual_seed(1)
        PT.run_rollout_arena([1, 2, 3, 4], piv1.clone(), forward_fn=fwd,
                             glue_rows_by_root=glue,
                             rope_base_by_root=[5] * R, device="cpu",
                             workspace=ws, **kw)
        # 2회차: 재사용 vs fresh 비교
        torch.manual_seed(2)
        ar_a, _, _ = PT.run_rollout_arena(
            [9, 8, 7, 6], piv2.clone(), forward_fn=fwd,
            glue_rows_by_root=glue, rope_base_by_root=[5] * R,
            device="cpu", workspace=ws, **kw)
        torch.manual_seed(2)
        ar_b, _, _ = PT.run_rollout_arena(
            [9, 8, 7, 6], piv2.clone(), forward_fn=fwd,
            glue_rows_by_root=glue, rope_base_by_root=[5] * R,
            device="cpu", workspace=None, **kw)
        pa, pb = ar_a.to_pool(R), ar_b.to_pool(R)
        self.assertEqual(pa.n, pb.n)
        n = pa.n
        for fld in ("tok", "parent_idx", "root", "cell", "sib_order"):
            self.assertEqual(getattr(pa, fld)[:n].tolist(),
                             getattr(pb, fld)[:n].tolist(), fld)
