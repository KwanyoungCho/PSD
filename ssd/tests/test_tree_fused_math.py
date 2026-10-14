import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ssd.engine.helpers.p2_tree import _arena_fanout_global
from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
from ssd.engine.helpers.tree_fused_math import pack_mask, fanout


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class FusedTreeMathTests(unittest.TestCase):
    @torch.inference_mode()
    def test_small_width_parallel_insert_matches_serial_with_fixed_proposals(self):
        from tests.test_p2_executor_parity import TestExecutorModuleParity
        fixture = TestExecutorModuleParity()
        for policy in ('level', 'dynamic'):
            ex, cfg, page, vocab = fixture._mk('cuda:0', policy=policy)
            ctx = page + 21
            bucket = (ctx + page - 1) // page
            ex.prepare_bucket(bucket)
            fixture._fill_inputs(ex, page, ctx)
            ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                       dtype=torch.long, device=ex.dev)
            ex.parity_noise = [torch.empty(ex.W, vocab, device=ex.dev).exponential_()
                               for _ in range(ex.F)]
            names = ('view_tok', 'view_par', 'view_sib', 'view_rawq',
                     'view_pcell', 'out_valid', 'out_pq_ref', 'out_pq_cells',
                     'out_u_valid', 'cell_logits')
            with patch.dict(os.environ, SSD_TREE_PARALLEL_INSERT='0'):
                ex.model.cache.zero_()
                ex.run_once(bucket)
            expected = {name: getattr(ex, name).clone() for name in names}
            expected_cache = ex.model.cache.clone()
            with patch.dict(os.environ, SSD_TREE_PARALLEL_INSERT='1'):
                ex.model.cache.zero_()
                ex.run_once(bucket)
                for name, value in expected.items():
                    self.assertTrue(torch.equal(value, getattr(ex, name)), (policy, name))
                self.assertTrue(torch.equal(expected_cache, ex.model.cache))
                graph = ex.capture(bucket)
                ex.model.cache.zero_()
                graph.replay()
                for name, value in expected.items():
                    self.assertTrue(torch.equal(value, getattr(ex, name)), (policy, name, 'replay'))

    @torch.inference_mode()
    def test_mask_exact_bytes_variable_width_prefix_words_and_replay(self):
        torch.manual_seed(71);dev=torch.device('cuda')
        for W,canvas,prior in [(3,256,0),(10,256,25),(22,768,88),(48,1024,129)]:
            roots=7;nodes=91;gwmax=19;words=3
            ar=SimpleNamespace(root=torch.randint(roots,(nodes,),device=dev),
                anc_bits=torch.randint(2**62,(nodes,words),device=dev),anc_words=words)
            sel=torch.randint(nodes,(W,),device=dev)
            valid=torch.rand(W,device=dev)>.2
            ex=SimpleNamespace(arena=ar,_sel={1:(sel,valid)},round_widths=(W,W),
                round_offsets=(0,prior),in_prefix_len=torch.tensor([37],device=dev),
                in_glue_w=torch.tensor([11],device=dev),
                in_glue=torch.randint(2,(roots,gwmax),dtype=torch.uint8,device=dev),
                lane_w=torch.arange(W,device=dev),dev=dev)
            wr=SimpleNamespace(_canvas_cols=canvas,_custom_mask_buf=torch.empty(W*canvas//8,dtype=torch.uint8,device=dev))
            with patch.dict(os.environ,SSD_TREE_FUSED_MATH='0'):
                P2TreeExecutor._pack_row_mask(ex,wr,1)
            expected=wr._custom_mask_buf.clone()
            pack_mask(ex,wr,1)
            self.assertTrue(torch.equal(expected,wr._custom_mask_buf),(W,canvas,prior))
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):pack_mask(ex,wr,1)
            ex.in_prefix_len.fill_(59);ex.in_glue_w.fill_(7);valid.logical_not_()
            with patch.dict(os.environ,SSD_TREE_FUSED_MATH='0'):
                P2TreeExecutor._pack_row_mask(ex,wr,1)
            expected=wr._custom_mask_buf.clone()
            graph.replay()
            self.assertTrue(torch.equal(expected,wr._custom_mask_buf))

    @torch.inference_mode()
    def test_fanout_exact_round_robin_with_ties_invalid_rows_and_reserve(self):
        torch.manual_seed(83);dev=torch.device('cuda')
        for W in (3,10,22,48):
            for seed in range(12):
                R=7
                # Quantized priorities deliberately create stable-sort ties.
                ar=SimpleNamespace(root=torch.randint(R,(128,),device=dev),
                    logpri=torch.randint(-3,1,(128,),device=dev).double(),device=dev)
                selected=torch.randperm(128,device=dev)[:W]
                valid=torch.rand(W,device=dev)>.3
                remaining=torch.randint(0,17,(R,),device=dev)
                future=seed%8;C=1+seed%3
                expected=_arena_fanout_global(ar,selected,valid,remaining,C,R,future)
                actual=fanout(ar,selected,valid,remaining,C,future)
                self.assertTrue(torch.equal(expected,actual),(W,seed,expected.tolist(),actual.tolist()))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=fanout(ar,selected,valid,remaining,C,future)
        remaining.fill_(2);valid.logical_not_();graph.replay()
        self.assertTrue(torch.equal(actual,_arena_fanout_global(ar,selected,valid,remaining,C,R,future)))


if __name__=='__main__':unittest.main()
