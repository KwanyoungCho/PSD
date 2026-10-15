"""Native last/frontier configuration, q-only scores and execution contracts."""
import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ssd.config import Config
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.helpers.tree_expansion_config import resolve_tree_expansion
from ssd.engine.helpers.tree_expansion_policy import TreeExpansionPolicy


def policy(mode='last', phase='p2', **overrides):
    cfg = SimpleNamespace(duet_tree_expansion=mode, **overrides)
    ex = SimpleNamespace(cfg=cfg, phase=phase, policy='dynamic',
                         W=8, R=8, F=8, NV=192, C=3)
    with contextlib.redirect_stdout(io.StringIO()):
        return TreeExpansionPolicy(ex)


class Configuration(unittest.TestCase):
    def test_default_and_phase_overrides(self):
        self.assertEqual(Config.duet_tree_expansion, 'last')
        self.assertEqual(Config.duet_tree_c_tensor, 3)
        self.assertEqual(Config.__new__(Config).duet_tree_proxy_threshold, 0.)
        self.assertEqual(Config.__new__(Config).duet_tree_conf_threshold, 0.)
        cfg = SimpleNamespace(duet_tree_expansion='frontier',
                              duet_p1_tree_expansion='last')
        self.assertEqual(resolve_tree_expansion(cfg, 'p1'), 'last')
        self.assertEqual(resolve_tree_expansion(cfg, 'p2'), 'frontier')
        self.assertEqual(resolve_tree_expansion(None, 'p1'), 'last')

    def test_rejects_unknown_values(self):
        for value in ('level', 'q_path', 'reach', '', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_tree_expansion(SimpleNamespace(duet_tree_expansion=value), 'p1')
        with self.assertRaises(ValueError):
            resolve_tree_expansion(SimpleNamespace(duet_p2_tree_expansion='typo'), 'p2')
        with self.assertRaises(ValueError):
            resolve_tree_expansion(None, 'ps')

    def test_bad_config_fails_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, 'last\\|frontier'):
            Config(model='/nonexistent', duet_tree_expansion='typo')
        with self.assertRaisesRegex(ValueError, 'requires dynamic'):
            Config(model='/nonexistent', duet_p1_tree_policy='on',
                   duet_p1_tree_allocation_policy='backbone', duet_tree_expansion='frontier')

    def test_phase_resolution_without_calibration_io(self):
        for phase in ('p1', 'p2'):
            for mode in ('last', 'frontier'):
                with self.subTest(phase=phase, mode=mode), \
                     patch.dict(os.environ, {'SSD_TREE_EXPANSION_POLICY': 'q_path'}), \
                     patch('pathlib.Path.read_bytes', side_effect=AssertionError('calibration read')):
                    p = policy(mode, phase)
                self.assertEqual(p.frontier, mode == 'frontier')
                self.assertEqual(p.mode, 'q_path')
                self.assertIsNone(p.alpha)
                self.assertIsNone(p.gain)

    def test_stale_calibration_environment_fails_loudly(self):
        with patch.dict(os.environ, {'SSD_TREE_EXPANSION_POLICY': 'reach_gain_frontier',
                                    'SSD_TREE_LEGACY_CALIBRATION': '0'}):
            with self.assertRaisesRegex(ValueError, 'calibration-free'):
                policy()
            with self.assertRaisesRegex(ValueError, 'calibration-free'):
                Config(model='/nonexistent', duet_enabled=True)

    def test_phase_overrides_survive_draft_config_reconstruction(self):
        from test_blackwell_review import make_config
        cfg, draft = make_config(duet_p1_tree_policy='on',
                                 duet_tree_expansion='frontier',
                                 duet_p1_tree_expansion='last')
        for config in (cfg, draft):
            self.assertEqual(resolve_tree_expansion(config, 'p1'), 'last')
            self.assertEqual(resolve_tree_expansion(config, 'p2'), 'frontier')

    def test_q_path_is_root_weight_times_raw_q_not_conditional_wor(self):
        raw = torch.tensor([[.5, .3, .2], [.1, .6, .3]], dtype=torch.float64)
        before = raw.clone()
        parent = torch.tensor([.8, .2], dtype=torch.float64).log()
        indices = torch.tensor([0, 0, 0, 1, 1, 1])
        for mode in ('last', 'frontier'):
            with patch.dict(os.environ, {'SSD_TREE_EXPANSION_POLICY': 'q_path'}):
                p = policy(mode)
            got = p.priority(parent, indices, raw).exp()
            torch.testing.assert_close(got, (parent.exp()[:, None] * raw).flatten())
            torch.testing.assert_close(raw, before, rtol=0, atol=0)


class Selection(unittest.TestCase):
    def test_full_cpu_rollout_and_arena_match_for_both_modes(self):
        import numpy as np
        from test_p2_tree_alloc import TestArenaParity
        for frontier in (False, True):
            for seed in (7, 23):
                with self.subTest(frontier=frontier, seed=seed):
                    cpu, arena, cpu_inputs, arena_inputs = TestArenaParity()._mk(W=4, F=4)
                    args = dict(policy='dynamic', W=4, F_total=4, c_tensor=3, nv=48,
                        beta=.5, depth_cap=4, temps=torch.full((4,), .7),
                        glue_rows_by_root=np.ones((2, 3), dtype=np.uint8),
                        rope_base_by_root=[20, 30], K_glue=2, context_len=80,
                        fanout_policy='ctensor', all_frontier=frontier)
                    roots, weights = [10, 11], torch.tensor([.8, .2])
                    torch.manual_seed(seed)
                    pool, _, q1 = pt.run_rollout(roots, weights, forward_fn=cpu, **args)
                    torch.manual_seed(seed)
                    ar, _, q2 = pt.run_rollout_arena(roots, weights, forward_fn=arena,
                                                   device='cpu', **args)
                    other = ar.to_pool(2)
                    self.assertEqual(pool.n, other.n)
                    for name in ('tok', 'depth', 'root', 'sib_order', 'parent_idx', 'parent_cell'):
                        torch.testing.assert_close(getattr(pool, name)[:pool.n],
                                                   getattr(other, name)[:other.n])
                    torch.testing.assert_close(q1, q2)
                    for (ids1, rope1, mask1), (ids2, rope2, mask2) in zip(cpu_inputs, arena_inputs):
                        torch.testing.assert_close(ids1, ids2)
                        torch.testing.assert_close(rope1, rope2)
                        self.assertEqual(mask1.tobytes(), mask2.tobytes())

    def test_serving_keeps_generation_prefix_without_score_rerank(self):
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        service = BatchedDuetDraft.__new__(BatchedDuetDraft)
        service.entries = {}
        service.cfg = SimpleNamespace(duet_p1_tree_policy='on', duet_p1_tree_verify_nodes=4)
        arena = SimpleNamespace(NV=6, out_valid=torch.tensor([6]),
            view_tok=torch.tensor([[10, 11, 12, 13, 14, 15]]),
            view_par=torch.tensor([[-1, -1, 0, 0, 1, 2]]),
            view_sib=torch.tensor([[0, 1, 0, 1, 0, 0]]),
            view_pcell=torch.tensor([[0, 0, 1, 1, 2, 3]]),
            cell_logits=torch.randn(4, 20))
        service._save_entries(arena, 1, torch.tensor([0]), torch.tensor([5]), 1, True)
        entry = service.entries[1, 0, 5]
        self.assertEqual(entry['tokens'], [10, 11, 12, 13])
        self.assertEqual(entry['parents'], [-1, -1, 0, 0])
        self.assertEqual(entry['siblings'], [0, 1, 0, 1])
        self.assertEqual(entry['cells'], [0, 0, 1, 1])

    def test_cpu_and_arena_selection_match_independent_oracle(self):
        # Multiple depths, an expanded high-score node, ties and zero-cap roots.
        depths = [0, 0, 1, 1, 2, 2, 1, 3, 2]
        roots = [0, 1, 0, 1, 0, 1, 0, 0, 0]
        scores = [0., 0., -.1, -.2, -1., -.5, -.1, -.01, -.01]
        states = [1, 1, 0, 0, 0, 0, 0, 0, 1]
        pool = pt.TreePool(capacity=16)
        for i, (depth, root, score, state) in enumerate(zip(depths, roots, scores, states)):
            pool.add(i, -1, -1, depth, root, 0, score, .5)
            pool.state[i] = state
        ar = pt.TreeArena(16, 'cpu', max_cells=16)
        ar.n.fill_(len(depths))
        ar.valid[:len(depths)] = True
        for name, values in (('depth', depths), ('root', roots),
                             ('logpri', scores), ('state', states)):
            getattr(ar, name)[:len(values)] = torch.tensor(values)
        for frontier in (False, True):
            for rem in ([20, 20], [2, 0], [0, 0]):
                with self.subTest(frontier=frontier, remaining=rem):
                    left = torch.tensor(rem)
                    quota = [max(1, r-1) if r > 0 else 0 for r in rem]
                    candidates = [i for i, d in enumerate(depths)
                                  if states[i] == 0 and d < 3
                                  and (d <= 2 if frontier else d == 2)]
                    candidates.sort(key=lambda i: (-scores[i], i))
                    expected = []
                    for i in candidates:
                        if quota[roots[i]]:
                            expected.append(i)
                            quota[roots[i]] -= 1
                        if len(expected) == 3:
                            break
                    cpu = pt.select_nodes_global(pool, 3, 2, 3, left, 1,
                                                 all_frontier=frontier)
                    sel, valid = pt._arena_select_global(ar, 3, 2, 3, left, 1, 2,
                                                         all_frontier=frontier)
                    self.assertEqual(cpu, expected)
                    self.assertEqual(sel[valid].tolist(), expected)

    def test_default_last_matches_explicit_last(self):
        ar = pt.TreeArena(8, 'cpu')
        ar.n.fill_(3); ar.valid[:3] = True
        ar.depth[:3] = torch.tensor([0, 1, 2])
        ar.root[:3] = 0
        ar.logpri[:3] = torch.tensor([0., -.1, -1.])
        args = (ar, 2, 2, 4, torch.tensor([8]), 1, 1)
        for a, b in zip(pt._arena_select_global(*args),
                        pt._arena_select_global(*args, all_frontier=False)):
            torch.testing.assert_close(a, b)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativePhaseGraphs(unittest.TestCase):
    @torch.inference_mode()
    def test_both_phases_modes_keep_q_topology_and_capture_parity(self):
        import test_p2_executor_parity as fixture
        from ssd.engine.helpers.p1_tree import P1TreeExecutor
        f = fixture.TestExecutorModuleParity()
        for mode in ('last', 'frontier'):
            for phase in ('p1', 'p2'):
                with self.subTest(mode=mode, phase=phase), \
                     patch.dict(os.environ, {'SSD_TREE_EXPANSION_POLICY': 'q_path'}), \
                     patch.object(fixture._MiniCfg, 'duet_tree_expansion', mode, create=True):
                    ex, cfg, page, v = f._mk('cuda', policy='dynamic')
                    if phase == 'p1':
                        ex = P1TreeExecutor(ex.model, ex.model.logits_fn, cfg, 'cuda',
                            page, 8, v, 4, 2, 64, context_bucket=3,
                            materialize_backbone_logits=False)
                    self.assertEqual(ex.expansion_policy.frontier, mode == 'frontier')
                    ex.prepare_bucket(2); f._fill_inputs(ex, page, page+21)
                    gen = torch.Generator().manual_seed(233)
                    ex.parity_noise = [torch.empty(w, v).exponential_(1, generator=gen).cuda()
                                       for w in ex.round_widths]
                    ex.model.cache.zero_(); ex.run_once(2)
                    names = ('view_tok', 'view_par', 'view_sib', 'view_rawq',
                             'view_pcell', 'out_valid', 'out_pq_ref')
                    expected = {k: getattr(ex, k).clone() for k in names}
                    for root in range(ex.R):
                        n = int(ex.out_valid[root])
                        self.assertLessEqual(n, ex.NV)
                        if not n:
                            continue
                        par = ex.view_par[root, :n].tolist()
                        sib = ex.view_sib[root, :n].tolist()
                        seen = {}
                        for j, (parent, sibling) in enumerate(zip(par, sib)):
                            self.assertLess(parent, j)
                            self.assertEqual(sibling, seen.get(parent, 0))
                            seen[parent] = sibling+1
                        cells = ex.view_pcell[root, :n]
                        q = pt.q_probs_from_logits(ex.cell_logits[cells], ex.in_temps[:1].expand(n))
                        raw = q.gather(1, ex.view_tok[root, :n, None])[:, 0]
                        torch.testing.assert_close(raw, ex.view_rawq[root, :n], rtol=2e-5, atol=1e-6)
                    ex.model.cache.zero_(); ex.capture(2)
                    ex.model.cache.zero_(); ex.replay(2)
                    for k, value in expected.items():
                        torch.testing.assert_close(value, getattr(ex, k), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
