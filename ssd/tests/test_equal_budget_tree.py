"""Independent small-tree tests for the matched-generation experiment."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import torch
from mlsys_equal_budget_adapter import select_roots, configure
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.helpers.tree_expansion_policy import TreeExpansionPolicy


class RootBudget(unittest.TestCase):
    def test_top_roots_matches_scalar_pairs(self):
        t=torch.tensor([[4,5,6,7],[2,3,4,5]])
        s=torch.tensor([[.2,.7,.3,.7],[.8,.1,.1,0.]])
        ci=torch.tensor([[0,0,1,1]])
        a,b,c=select_roots(t,s,ci,3)
        self.assertEqual(a.tolist(),[[5,7,6],[2,3,4]])
        self.assertEqual(c.tolist(),[[0,1,1],[0,0,1]])
        self.assertEqual(select_roots(t[:,:1],s[:,:1],ci[:,:1],3)[1][:,1:].sum(),0)

    def test_no_calibration_default(self):
        with patch.dict(os.environ,clear=False):
            os.environ.pop('SSD_TREE_EXPANSION_POLICY',None)
            p=TreeExpansionPolicy(None)
        self.assertEqual(p.mode,'q_path');self.assertIsNone(p.alpha);self.assertIsNone(p.gain)

    def test_nonbinding_generation_capacity(self):
        for c in (1,3,8):
            kw=dict(max_num_seqs=8,duet_phase1_k=8,duet_phase2_k=8,duet_tree_c_tensor=c)
            with patch.dict(os.environ,{'DUET_EQUAL_ROOTS':'8'}):configure(kw)
            self.assertEqual(kw['duet_p2_budget'],8)
            self.assertEqual(kw['duet_p1_tree_max_nodes'],64*c)

    def test_last_and_frontier_have_different_eligibility(self):
        ar=pt.TreeArena(8,'cpu',max_cells=8)
        ar.n.fill_(4);ar.valid[:4]=True;ar.state[:4]=torch.tensor([1,0,0,0])
        ar.depth[:4]=torch.tensor([0,1,2,2]);ar.root[:4]=0
        ar.logpri[:4]=torch.tensor([0.,-.1,-2.,-3.])
        for frontier,expected in ((False,2),(True,1)):
            sel,valid=pt._arena_select_global(ar,1,2,8,torch.tensor([100]),5,1,all_frontier=frontier)
            self.assertTrue(valid.item());self.assertEqual(sel.item(),expected)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class LargeExpansion(unittest.TestCase):
    @torch.inference_mode()
    def test_eight_rounds_c1_c3_c8_preserve_q_and_graph_outputs(self):
        if __package__:
            from . import test_p2_executor_parity as f
        else:
            import test_p2_executor_parity as f
        from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
        fixture=f.TestExecutorModuleParity()
        for c in (1,3,8):
            for frontier in (False,True):
                with self.subTest(c=c,frontier=frontier),patch.dict(os.environ,{'SSD_TREE_EXPANSION_POLICY':'q_path'}):
                    old,cfg,page,v=fixture._mk('cuda',policy='dynamic')
                    cfg.duet_tree_c_tensor=c
                    cfg.duet_tree_proxy_threshold=0.;cfg.duet_tree_conf_threshold=0.
                    ex=P2TreeExecutor(old.model,old.model.logits_fn,cfg,'cuda',page,8,v,4,2,64,
                        width=8,root_count=8,depth=8,max_nodes=64*c,glue_width=9,
                        materialize_backbone_logits=False)
                    ex.expansion_policy.frontier=frontier;ex.debug_buffers_enabled=True
                    ex._finalize_outputs=lambda:None
                    ex.prepare_bucket(2);fixture._fill_inputs(ex,page,page+21)
                    gen=torch.Generator().manual_seed(21)
                    ex.parity_noise=[torch.empty(8,v).exponential_(1,generator=gen).cuda() for _ in range(8)]
                    ex.model.cache.zero_();ex.run_once(2)
                    names=('view_tok','view_par','view_sib','view_rawq','view_pcell','out_valid')
                    expected={k:getattr(ex,k).clone() for k in names}
                    self.assertEqual(ex.round_widths,(8,)*8)
                    self.assertEqual(int(ex.out_valid.sum()),64*c)
                    for r in range(8):
                        n=int(ex.out_valid[r]);tok=ex.view_tok[r,:n];cells=ex.view_pcell[r,:n]
                        q=pt.q_probs_from_logits(ex.cell_logits[cells],torch.full((n,),.8,device='cuda'))
                        torch.testing.assert_close(q.gather(1,tok[:,None])[:,0],ex.view_rawq[r,:n],rtol=2e-5,atol=1e-6)
                        par=ex.view_par[r,:n].tolist();sib=ex.view_sib[r,:n].tolist()
                        children={}
                        for j,(p,s) in enumerate(zip(par,sib)):
                            self.assertLess(p,j);children.setdefault(p,[]).append(s)
                        for values in children.values():self.assertEqual(values,list(range(len(values))))
                        if c==1:self.assertEqual(par,list(range(-1,7)))
                    ex.model.cache.zero_();ex.capture(2);ex.model.cache.zero_();ex.replay(2)
                    for k,x in expected.items():torch.testing.assert_close(x,getattr(ex,k),rtol=1e-5,atol=1e-6)
                    del ex,old
                    torch.cuda.empty_cache()

if __name__=='__main__':unittest.main()
