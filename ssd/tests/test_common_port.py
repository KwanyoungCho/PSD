"""Common paper-branch correctness contracts, independent of research policies."""
import os, unittest
from types import SimpleNamespace
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.helpers.proxy_temperature import probabilities

class CommonPort(unittest.TestCase):
    def test_legacy_prefix_ignores_realized_q_and_stale_cache(self):
        for phase in (1,2):
            for score in (.001,.999):
                stub = SimpleNamespace(config=SimpleNamespace(
                    duet_p1_tree_max_nodes=3, duet_p2_tree_max_nodes=3,
                    duet_p1_tree_verify_nodes=2, duet_p2_tree_verify_nodes=2,
                    duet_tree_wire_nodes=2), hf_config=SimpleNamespace(vocab_size=8),
                    device=torch.device('cpu'), _p1_rerank_cache={'invalid_old_cache':True})
                stub._tree_compact_view = {k:torch.zeros(1,2,dtype=torch.int64)
                    for k in ('tok','parent_local','sib_order','parent_q_ref','parent_q_cells')}
                stub._tree_compact_view.update(raw_q=torch.zeros(1,2),valid=torch.zeros(1,dtype=torch.int64),
                    u_valid=torch.zeros(1,dtype=torch.int64),selected_old=torch.zeros(2,dtype=torch.int64))
                views=dict(tok=torch.tensor([[0,1,2]]),parent_local=torch.tensor([[-1,-1,0]]),
                    sib_order=torch.tensor([[0,1,0]]),raw_q=torch.tensor([[.6,.25,score]]),
                    valid=torch.tensor([3]),parent_q_ref=torch.tensor([[0,0,1]]),
                    parent_q_cells=torch.tensor([[0,1,-1]]),u_valid=torch.tensor([2]),
                    cell_logits=torch.randn(2,8))
                served,*_=DraftRunner._rerank_tree_hit_view(stub,views,0,phase)
                self.assertEqual(served['tok'][0].tolist(),[0,1])
                self.assertEqual(served['parent_local'][0].tolist(),[-1,-1])
                self.assertEqual(served['parent_q_ref'][0].tolist(),[0,0])
                DraftRunner._precompute_p1_rerank_views(stub,views,1)
                self.assertIsNone(stub._p1_rerank_cache)
    def test_large_vocab_wire_and_old_version_rejection(self):
        for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            ids=torch.tensor([0,32767,32768,128255,151935,2**32-1],device=dev)
            p=torch.tensor([1.,.3,.02,1e-3,1e-5,1e-6],device=dev)
            packed=pt.pack_piv(ids,p); actual,score=pt.unpack_piv(packed)
            self.assertTrue(torch.equal(ids,actual)); self.assertEqual(packed.element_size(),8)
            torch.testing.assert_close(score,p,rtol=.00012,atol=1e-10)
            with self.assertRaises(ValueError):pt.unpack_piv(torch.tensor([1<<31],device=dev))

    def test_temperature_probabilities_independent_reference(self):
        torch.manual_seed(103)
        for dtype in (torch.float16,torch.bfloat16):
            logits=torch.randn(4,5,97,dtype=dtype)
            ts=[0.,.3,.7,1.]
            actual=probabilities(logits,ts)
            expect=torch.stack([torch.softmax(row.float()/(t if t>0 else 1.),dim=-1) for row,t in zip(logits,ts)])
            torch.testing.assert_close(actual,expect,rtol=0,atol=0)
            self.assertTrue(torch.equal(probabilities(logits,1.),logits.float().softmax(-1)))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_live_chain_tree_graph_temperature_and_t0_ranking(self):
        torch.manual_seed(82)
        dev=torch.device('cuda'); K,V=4,129
        ex=torch.randn(K+1,V,device=dev,dtype=torch.float16)
        q=torch.randn(K,V,device=dev,dtype=torch.float16)
        tok=torch.tensor([1,2,3,4],device=dev)
        # Graph inputs must remain FP32 even in the engine's FP16 default context.
        saved=torch.get_default_dtype();torch.set_default_dtype(torch.float16)
        try:
            chain=pt.ChainProxyCUDAGraph(K,V,8,10,True,torch.float16,dev)
            tree=pt.TreeProxyCUDAGraph(K,V,10,K,torch.float16,dev,top_k=8)
        finally:torch.set_default_dtype(saved)
        tree.prepare_topology([-1,-1,0,1],[0,1,0,0])
        self.assertEqual(chain.tt.dtype,torch.float32); self.assertEqual(tree.tt.dtype,torch.float32)
        for a,b in [(1.,1.),(.7,.3),(0.,0.),(.3,1.),(1.,1.)]:
            for graph,is_tree in [(chain,False),(tree,True)]:
                out=tuple(x.clone() for x in graph.replay(ex,q,tok,a,b))
                fn = (lambda ex,ql,**kw: pt.tree_proxy_candidates_fixed(ex,ql,tok,tree.topology,10,K,8,**kw)) if is_tree else (lambda ex,ql,**kw: pt.chain_proxy_candidates_fixed(ex,ql,tok,8,10,True,**kw))
                direct=fn(ex,q,target_temps=a,draft_temps=b)
                for x,y in zip(out,direct):torch.testing.assert_close(x,y,rtol=0,atol=0)
                # Independent scalar division can differ by FP32 roundoff and
                # move a score by one wire quantization bin; IDs stay equal.
                ref=fn(ex.float()/(a or 1.),q.float()/(b or 1.))
                self.assertTrue(torch.equal(out[0],ref[0]))
                ids,score=pt.unpack_piv(out[1]);ref_ids,ref_score=pt.unpack_piv(ref[1])
                self.assertTrue(torch.equal(ids,ref_ids))
                torch.testing.assert_close(score,ref_score,rtol=.00023,atol=1e-10)
                torch.testing.assert_close(out[2],ref[2],rtol=1e-6,atol=1e-7)

if __name__=='__main__':unittest.main()
