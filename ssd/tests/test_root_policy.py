import os
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import unittest
import torch
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ssd.engine.helpers.root_policy import chain_candidates, tree_candidates
from ssd.engine.helpers.batch_tree_sampling import pack_topologies, ladder, BatchedTreeProxy
from ssd.engine.helpers.batched_proxy import BatchedChainProxyCUDAGraph
from ssd.engine.helpers.p2_tree import unpack_piv


POLICY=dict(source='complement',normalization='full',overlap_mix=.25)


class RootPolicyTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA proxy side-stream ordering')
    @torch.inference_mode()
    def test_proxy_side_stream_preserves_graph_outputs_and_reuse(self):
        from ssd.engine.helpers.tree_proxy_stream import TreeProxySideStream
        graph=BatchedTreeProxy(2,4,257,torch.float32,'cuda',8,4,8,policy=POLICY)
        torch.manual_seed(942)
        e=torch.randn(2,5,257,device='cuda');q=torch.randn(2,4,257,device='cuda');q[:,1]=q[:,0]
        tok=torch.tensor([[5,7,1,2],[9,8,0,0]],device='cuda')
        side=TreeProxySideStream('cuda')
        for temp in (.7,.4,1.):
            graph.prepare([[-1,-1,0,1],[-1,0]],[[0,1,0,0],[0,0]],tok,q,[temp,temp],[temp,temp])
            expected=tuple(x.clone() for x in graph.replay(e))
            actual=[]
            def callback(logits,batch):
                torch.cuda._sleep(100000)
                actual.extend(x.clone() for x in graph.replay(logits))
            side.launch(e,2,callback)
            unrelated=(e.square()+q.sum()).sum()
            side.finish()
            # The caller may immediately reuse inputs on its own stream.
            e.add_(.2);q.mul_(.99)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(unrelated).item())

    def test_trim_preserves_root_ranking_and_exact_correction(self):
        from ssd.engine.helpers.batch_tree_sampling import verify_batch
        torch.manual_seed(207)
        e=torch.randn(2,5,127);q=torch.randn(2,4,127);q[:,1]=q[:,0]
        tokens=torch.tensor([[5,7,1,2],[9,8,0,0]])
        topo=pack_topologies([[-1,-1,0,1],[-1,0]],[[0,1,0,0],[0,0]],4,'cpu')
        for source in ('proxy','complement','residual'):
            outputs=[]
            for trim in ('0','1'):
                with patch.dict(os.environ,SSD_TREE_LADDER_TRIM=trim):
                    outputs.append(tree_candidates(e,q,tokens,topo,8,4,8,.7,.7,
                        source=source,normalization='full',overlap_mix=.25))
            for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)
        outputs=[]
        for trim in ('0','1'):
            with patch.dict(os.environ,SSD_TREE_LADDER_TRIM=trim):
                torch.manual_seed(841)
                outputs.append(verify_batch(e,q,tokens,topo,[.7,.7],[.7,.7]))
        for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'requires the final residual'):
            ladder(tokens,e.softmax(-1),q.softmax(-1),topo,4,exact=True,need_residual=False,trim=True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA graph trim parity')
    @torch.inference_mode()
    def test_trim_captured_proxy_and_exact_walk_are_bit_identical(self):
        from ssd.engine.helpers.batch_tree_sampling import BatchedTreeAccept
        torch.manual_seed(291)
        e=torch.randn(2,5,257,device='cuda');q=torch.randn(2,4,257,device='cuda');q[:,1]=q[:,0]
        tokens=torch.tensor([[5,7,1,2],[9,8,0,0]],device='cuda')
        outputs=[]
        for trim in ('0','1'):
            with patch.dict(os.environ,SSD_TREE_LADDER_TRIM=trim):
                proxy=BatchedTreeProxy(2,4,257,torch.float32,'cuda',8,4,8,policy=POLICY)
                proxy.prepare([[-1,-1,0,1],[-1,0]],[[0,1,0,0],[0,0]],tokens,q,[.7,.7],[.7,.7])
                root=tuple(x.clone() for x in proxy.replay(e))
                graph=BatchedTreeAccept(proxy,None,1,False)
                torch.manual_seed(431)
                exact=tuple(x.clone() for x in graph.replay(e,[.7,.7],[.7,.7]))
                outputs.append(root+exact)
        for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_gain_curve_prefix_for_narrower_sibling_width(self):
        from ssd.engine.helpers.tree_expansion_policy import TreeExpansionPolicy
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gain.json'
            path.write_text(json.dumps({'curves':{'1':[0.,.5,.7,.8]}}))
            for width in (1,2,3):
                ex=SimpleNamespace(C=width,NV=4,phase='p1',policy='dynamic',dev='cpu',
                    cfg=SimpleNamespace(duet_p1_tree_policy='on',duet_p1_tree_verify_nodes=4))
                with patch.dict(os.environ,SSD_TREE_EXPANSION_POLICY='q_gain',
                                SSD_TREE_GAIN_CALIBRATION=str(path)):
                    self.assertIsNotNone(TreeExpansionPolicy(ex).gain)

    def test_against_scalar_reference_and_actual_temperature(self):
        e=torch.tensor([[[.8,.1,.06,.04],[.1,.6,.2,.1],[.3,.4,.2,.1]]]).log()
        q=torch.tensor([[[.5,.3,.1,.1],[.3,.4,.2,.1]]]).log()
        tok=torch.tensor([[0,1]])
        previous=None
        for temp in (.7,1.,.5):
            pe=(e.double()/temp).softmax(-1)[0].tolist()
            pq=(q.double()/temp).softmax(-1)[0].tolist()
            obs=[]; avg=[]
            for i,t in enumerate(tok[0]):
                obs.append(min(1,pe[i][t]/pq[i][t]))
                avg.append(sum(min(x,y) for x,y in zip(pe[i],pq[i])))
            def mass(a):return [1-a[0],a[0]*(1-a[1]),a[0]*a[1]]
            h=[.75*x+.25*y for x,y in zip(mass(obs),mass(avg))]
            expected=[]
            for i in range(3):
                score=[p*(1-pq[i][v]) if i<2 else p for v,p in enumerate(pe[i])]
                if i<2:score[tok[0,i]]=0
                total=sum(score)
                expected.extend((h[i]*s/total,i,v) for v,s in enumerate(score))
            expected=sorted(expected,reverse=True)[:5]
            pos,ids,values=chain_candidates(e,q,tok,torch.tensor([2]),4,5,
                False,temp,temp,**POLICY)
            self.assertEqual(list(zip(pos[0].tolist(),ids[0].tolist())),[(i,v) for _,i,v in expected])
            torch.testing.assert_close(values[0],torch.tensor([x for x,_,_ in expected]),rtol=2e-6,atol=1e-7)
            if previous is not None:self.assertFalse(torch.allclose(previous,values))
            previous=values

    def test_ragged_bonus_uses_actual_end_and_keeps_padded_token(self):
        e=torch.tensor([[[.5,.3,.2],[.8,.15,.05],[.1,.2,.7]]]).log()
        q=torch.tensor([[[.5,.3,.2],[.05,.15,.8]]]).log()
        tok=torch.tensor([[0,0]])
        pos,ids,values=chain_candidates(e,q,tok,torch.tensor([1]),3,3,False,**POLICY)
        self.assertEqual(pos.tolist(),[[1,1,1]])
        self.assertEqual(ids.tolist(),[[0,1,2]])
        torch.testing.assert_close(values,torch.tensor([[.8,.15,.05]]))

    def test_chain_reduction_mixed_temperatures(self):
        torch.manual_seed(5)
        e=torch.randn(2,5,37); q=torch.randn(2,4,37); tok=torch.randint(37,(2,4))
        topo=pack_topologies([[-1,0,1,2],[-1,0]],[[0]*4,[0]*2],4,'cpu')
        for source in ('complement','proxy'):
            policy=dict(POLICY,source=source)
            a=chain_candidates(e,q,tok,torch.tensor([4,2]),10,10,True,
                torch.tensor([.7,.5]),torch.tensor([.9,.6]),**policy)
            b=tree_candidates(e,q,tok,topo,10,4,10,
                torch.tensor([.7,.5]),torch.tensor([.9,.6]),**policy)
            torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[1],b[1])
            torch.testing.assert_close(a[2],b[2])

    def test_ordered_sibling_overlap_terminal_mass(self):
        p=torch.tensor([[[.4,.3,.3],[.2,.5,.3],[.6,.1,.3]]])
        q=torch.tensor([[[.8,.1,.1],[.8,.1,.1]]])
        topo=pack_topologies([[-1,-1]],[[0,1]],2,'cpu')
        _,term,_=ladder(torch.tensor([[0,1]]),p,q,topo,1,overlap_mix=.25)
        torch.testing.assert_close(term,torch.tensor([[0,.525,.475]]))
        torch.testing.assert_close(term.sum(1),torch.ones(1))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA graph contract')
    def test_model_default_dtype_does_not_round_live_temperature(self):
        from ssd.engine.helpers.p2_tree import ChainProxyCUDAGraph
        before=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            graph=ChainProxyCUDAGraph(2,37,5,5,True,torch.bfloat16,'cuda',policy=POLICY)
            self.assertEqual(graph.tt.dtype,torch.float32)
            e=torch.randn(3,37,device='cuda');q=torch.randn(2,37,device='cuda')
            tok=torch.tensor([1,3],device='cuda')
            actual=graph.replay(e,q,tok,.7,.61)
            expected=chain_candidates(e[None],q[None],tok[None],tok.new_tensor([2]),
                5,5,True,.7,.61,**POLICY)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b[0])
        finally:torch.set_default_dtype(before)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA graph contract')
    def test_graph_reads_live_temperatures_chain_and_tree(self):
        device='cuda';torch.manual_seed(71)
        e=torch.randn(2,5,257,device=device);q=torch.randn(2,4,257,device=device)
        tok=torch.randint(257,(2,4),device=device);vk=torch.tensor([4,2],device=device)
        graph=BatchedChainProxyCUDAGraph(2,4,257,10,10,True,torch.float32,device,policy=POLICY)
        tree=BatchedTreeProxy(2,4,257,torch.float32,device,10,4,10,policy=POLICY)
        for temps in ([.7,.5],[1.,.7],[0.,.3]):
            t=torch.tensor(temps,device=device)
            a=chain_candidates(e,q,tok,vk,10,10,True,t,t,**POLICY)
            b=graph.replay(e,q,tok,vk,t,t)
            for x,y in zip(a,b):torch.testing.assert_close(x,y)
            tree.prepare([[-1,0,1,2],[-1,0]],[[0]*4,[0]*2],tok,q,temps,temps)
            c=tree.replay(e)
            for x,y in zip(a[:2],c):torch.testing.assert_close(x,y)


if __name__=='__main__':unittest.main()
