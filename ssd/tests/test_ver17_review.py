"""Regressions for independent tree RNG, exact score wire and queue ownership."""
from collections import deque
from types import SimpleNamespace
import unittest
import torch
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.helpers.tree_rng import tree_rng_seed
from ssd.engine.scheduler import Scheduler
from ssd.engine.sequence import SequenceStatus


class SchedulerOwnership(unittest.TestCase):
    def make_scheduler(self, lengths, speculate=True):
        s = Scheduler.__new__(Scheduler)
        s.max_model_len = 2048
        s.max_num_seqs = len(lengths)
        s.max_num_batched_tokens = 16384
        s.speculate = speculate
        s.draft_async = False
        s.K = s.response_width = 8
        s.waiting = deque()
        rows = [SimpleNamespace(seq_id=i, num_tokens=n,
                                status=SequenceStatus.RUNNING)
                for i, n in enumerate(lengths)]
        s.running = deque(rows)
        self.allocations = {r.seq_id for r in rows}
        self.draft_allocations = self.allocations.copy()
        self.appended = []
        s.block_manager = SimpleNamespace(
            deallocate=lambda r: self.allocations.remove(r.seq_id),
            may_append=lambda r, n: self.appended.append(r.seq_id))
        s.draft_block_manager = SimpleNamespace(
            deallocate=lambda r: self.draft_allocations.remove(r.seq_id),
            may_append=lambda r, n: None)
        s.bms_can_append = lambda *args: True
        return s, rows

    def test_middle_guard_keeps_popped_and_unvisited_requests(self):
        s, rows = self.make_scheduler([20, 30, 2041, 40])
        done, kind = s.schedule()
        self.assertIsNone(kind)
        self.assertEqual(done, [rows[2]])
        self.assertEqual(list(s.running), [rows[0], rows[1], rows[3]])
        self.assertEqual(self.allocations, {0, 1, 3})
        self.assertEqual(self.draft_allocations, {0, 1, 3})
        self.assertEqual(rows[0].status, SequenceStatus.RUNNING)
        scheduled, kind = s.schedule()
        self.assertIs(kind, False)
        self.assertEqual(scheduled, [rows[0], rows[1], rows[3]])
        self.assertEqual(list(s.running), scheduled)

    def test_repeated_guards_never_lose_or_duplicate_selected_request(self):
        s, rows = self.make_scheduler([20, 2041, 2042, 30])
        self.assertEqual(s.schedule()[0], [rows[1]])
        self.assertEqual(s.schedule()[0], [rows[2]])
        self.assertEqual(s.schedule()[0], [rows[0], rows[3]])
        self.assertEqual(self.allocations, {0, 3})
        self.assertEqual(list(s.running), [rows[0], rows[3]])

    def test_non_speculative_guard_and_single_request(self):
        for lengths in ([20, 2048], [2048]):
            s, rows = self.make_scheduler(lengths, speculate=False)
            done, kind = s.schedule()
            self.assertIsNone(kind)
            self.assertEqual(done, [rows[-1]])
            self.assertEqual(list(s.running), rows[:-1])


class ExactScoreWire(unittest.TestCase):
    def devices(self):
        return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

    def test_exact_bits_zero_subnormal_tiny_and_large_token_ids(self):
        # Construct FP32 payloads from bits, including subnormals. Bitwise
        # transport must not run a float operation that flushes these to 0.
        bits = torch.tensor([0, 1, 10, 8388607, 8388608,
                             0x3089705f, 0x3eaaaaab, 0x3f800000], dtype=torch.int32)
        tok = torch.tensor([0, 2**32-1, 32768, 128255, 151935, 1, 2, 3])
        for dev in self.devices():
            score = bits.to(dev).view(torch.float32)
            packed = pt.pack_piv(tok.to(dev), score)
            actual_tok, actual_score = pt.unpack_piv(packed)
            self.assertTrue(torch.equal(actual_tok.cpu(), tok))
            self.assertTrue(torch.equal(actual_score.view(torch.int32).cpu(), bits))
            self.assertEqual(packed.numel()*packed.element_size(), tok.numel()*8)

    def test_no_clipping_or_rank_ties_for_tiny_positive_scores(self):
        for dev in self.devices():
            p = torch.tensor([1e-12, 1e-9, 1e-8, 1e-6, .1], device=dev)
            ids = torch.arange(5, device=dev)
            _, decoded = pt.unpack_piv(pt.pack_piv(ids, p))
            self.assertTrue(torch.equal(decoded, p))
            self.assertTrue(bool((decoded[1:] > decoded[:-1]).all()))

    def test_strided_and_low_precision_inputs(self):
        for dev in self.devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                score = torch.linspace(0, 1, 32, device=dev, dtype=dtype).reshape(4, 8).T
                ids = torch.arange(32, device=dev).reshape(4, 8).T
                token, actual = pt.unpack_piv(pt.pack_piv(ids, score))
                self.assertTrue(torch.equal(token, ids))
                self.assertTrue(torch.equal(actual, score.float()))

    def test_legacy_payload_rejected(self):
        for dev in self.devices():
            with self.assertRaisesRegex(ValueError, 'v3'):
                pt.unpack_piv(torch.tensor([5 | (1 << 48)], device=dev))

    def test_zero_score_has_no_budget(self):
        p = torch.tensor([.5, 1e-10, 0.])
        _, decoded = pt.unpack_piv(pt.pack_piv(torch.arange(3), p))
        for policy in ('dynamic', 'eagle', 'confidence'):
            b = pt.alloc_policy_root_budgets(decoded, policy, total=24, beta=.5, cap=8)
            self.assertEqual(int(b[-1]), 0)

    def test_selector_ignores_zeros_after_dedup_and_underfills_safely(self):
        from ssd.engine.draft_runner import DraftRunner
        for dev in self.devices():
            proxy = dict(chosen_pos=torch.tensor([[0, 1, 1, 2]], device=dev),
                         chosen_tok=torch.tensor([[10, 11, 12, 13]], device=dev),
                         chosen_piv=torch.tensor([[.4, 0., 1e-12, 0.]], device=dev))
            roots = torch.tensor([[[10], [-1], [-1]]], device=dev)
            tokens, fan, scores = DraftRunner._select_proxy_sourced_tokens_unified(
                proxy, roots, K_rank=2, total_budget=3)
            self.assertEqual(int((scores > 0).sum()), 1)
            self.assertEqual(tokens[scores > 0].item(), 12)
            self.assertEqual(int(fan.sum()), 3)  # physical shape includes padding
            self.assertTrue(torch.equal(tokens[scores == 0], torch.zeros_like(tokens[scores == 0])))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_captured_executor_preserves_tiny_root_prior_and_ignores_zero(self):
        from test_p2_executor_parity import TestExecutorModuleParity
        fixture=TestExecutorModuleParity()
        ex,cfg,page,v=fixture._mk('cuda:0',policy='dynamic')
        ex.prepare_bucket(2)
        fixture._fill_inputs(ex,page,page+21)
        ex.in_root_piv[-3:].copy_(torch.tensor([1e-12,1e-8,0.],device='cuda'))
        ex.capture(2)
        ex.replay(2)
        expected=ex.in_root_piv.log().double()
        torch.testing.assert_close(ex.arena.logpri[:ex.R],expected,rtol=0,atol=0)
        self.assertEqual(int(ex.out_valid[ex.R-1]),0)


class BatchedChildExclusion(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_ragged_forests_match_cpu_and_captured_topology_changes(self):
        from ssd.engine.helpers.batch_tree_roots import root_candidates
        from ssd.engine.helpers.tree_host_topology import context_topology
        torch.manual_seed(413)
        b,p,v,u=3,6,31,2
        logits=torch.randn(b,p,v)
        tokens=torch.randint(v,(b,p))
        parents=[[-1,-1,0,1],[-1,0],[]]
        lengths=torch.tensor([len(x)+1 for x in parents])
        vis=torch.zeros(b,p,p,dtype=torch.uint8)
        for i,par in enumerate(parents):vis[i,:lengths[i],:lengths[i]]=torch.as_tensor(context_topology(par)[2])
        temps=torch.tensor([.7,0.,1.2])
        inp=[x.cuda() for x in (logits,tokens,vis,lengths,temps)]
        root_candidates(*inp,u,None,3)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=root_candidates(*inp,u,None,3)
        for iteration in range(2):
            if iteration:
                parents[0]=[-1,0,1,2]
                vis[0].zero_();vis[0,:5,:5]=torch.as_tensor(context_topology(parents[0])[2])
                tokens.add_(3).remainder_(v);inp[1].copy_(tokens);inp[2].copy_(vis)
            graph.replay()
            ref=root_candidates(logits,tokens,vis,lengths,temps,u,None,3)
            for i,n in enumerate(lengths.tolist()):
                torch.testing.assert_close(out[0][i,:n*u].cpu(),ref[0][i,:n*u])
                torch.testing.assert_close(out[1][i,:n*u].cpu(),ref[1][i,:n*u],atol=2e-7,rtol=2e-6)
                self.assertFalse(bool(out[1][i,n*u:].any()))


class RNGDomains(unittest.TestCase):
    def test_domains_unique_stable_and_full_seed_bits_used(self):
        ids = [(phase, cb, lane) for phase in ('p1','p2')
               for cb in range(1,33) for lane in range(16)]
        values = [tree_rng_seed(2, *x) for x in ids]
        self.assertEqual(len(set(values)), len(values))
        self.assertEqual(values, list(reversed([tree_rng_seed(2,*x) for x in reversed(ids)])))
        self.assertNotEqual(tree_rng_seed(2,'p1',24), tree_rng_seed(2+2**31,'p1',24))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_real_executors_start_in_distinct_domains(self):
        from test_p2_executor_parity import _MiniCfg
        from ssd.engine.helpers.p1_tree import P1TreeExecutor
        from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
        torch.manual_seed(42)
        args = (None, None, _MiniCfg(), 'cuda', 64, 8, 128, 4, 2, 64)
        p1a = P1TreeExecutor(*args, context_bucket=6)
        p1b = P1TreeExecutor(*args, context_bucket=10)
        p2 = P2TreeExecutor(*args)
        gens = [ex.gen for ex in (p1a,p1b,p2)]
        self.assertEqual(len({g.initial_seed() for g in gens}), 3)
        self.assertTrue(all(g.initial_seed()!=torch.initial_seed() for g in gens))
        samples = [torch.empty(4096, device='cuda').exponential_(generator=g) for g in gens]
        self.assertTrue(all(not torch.equal(samples[i],samples[j]) for i in range(3) for j in range(i)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_uniform_two_token_joint_after_bucket_switch(self):
        n = 131072
        logits = torch.zeros(n,2,device='cuda')
        temp = torch.ones(n,device='cuda')
        def sample(g):
            return pt.tree_sample_wor(logits,temp,1,generator=g)[0].flatten()
        # Reproduce the old duplicated stream with the production sampler.
        g0 = torch.Generator(device='cuda').manual_seed(42)
        g1 = torch.Generator(device='cuda').manual_seed(42)
        a,b = sample(g0),sample(g1)
        self.assertTrue(torch.equal(a,b))
        g0.manual_seed(tree_rng_seed(42,'p1',6))
        g1.manual_seed(tree_rng_seed(42,'p1',10))
        a,b = sample(g0),sample(g1)
        joint = torch.bincount(2*a+b,minlength=4).float()/n
        self.assertTrue(bool((joint-.25).abs().lt(.01).all()),joint)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_graph_replay_advances_and_restores_each_domain(self):
        gens = [torch.Generator(device='cuda').manual_seed(tree_rng_seed(10,'p1',c)) for c in (6,24)]
        graphs, buffers, states = [], [], []
        for gen in gens:
            out = torch.empty(2048,device='cuda')
            state = gen.get_state().clone()
            out.exponential_(generator=gen)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(gen)
            with torch.cuda.graph(graph):
                out.exponential_(generator=gen)
            gen.set_state(state)
            graphs.append(graph);buffers.append(out);states.append(state)
        for graph in graphs:graph.replay()
        first = [x.clone() for x in buffers]
        self.assertFalse(torch.equal(*first))
        for graph in graphs:graph.replay()
        self.assertTrue(all(not torch.equal(x,y) for x,y in zip(first,buffers)))
        for gen,state in zip(gens,states):gen.set_state(state)
        for graph in graphs:graph.replay()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(first,buffers)))


if __name__ == '__main__':unittest.main()
