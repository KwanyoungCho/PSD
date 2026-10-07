"""Independent checks of batch tree wire geometry and residual verification."""
import unittest
import numpy as np
import torch
from ssd.engine.helpers.batch_tree_common import build_forward_inputs, restore_plan
from ssd.engine.helpers.batch_tree_sampling import pack_topologies, ladder, candidates, verify_batch
from ssd.engine.helpers.p2_tree import tree_proxy_candidates_fixed, unpack_piv

class Geometry(unittest.TestCase):
    def test_ancestors_positions_padding_and_pages(self):
        rows=[dict(tokens=[10,11,12,13,14],parents=[-1,-1,0,1],prefix=3,blocks=[7,2]),
              dict(tokens=[20,21,22],parents=[-1,0],prefix=1,blocks=[9,4])]
        x=build_forward_inputs(rows,5,4,2,4)
        np.testing.assert_array_equal(x['rope'].reshape(4,5)[0],[3,4,4,5,5])
        np.testing.assert_array_equal(x['slots'].reshape(4,5)[0],[31,8,9,10,11])
        self.assertTrue((x['slots'].reshape(4,5)[1,3:]==-1).all())
        self.assertTrue((x['slots'].reshape(4,5)[2:]==-1).all())
        mask=np.unpackbits(x['mask'],bitorder='little').reshape(4,5,8)
        np.testing.assert_array_equal(mask[0,3],[1,1,1,1,1,0,1,0])
        np.testing.assert_array_equal(mask[0,4],[1,1,1,1,0,1,0,1])
        self.assertTrue(mask[2,:,:3].all())
        self.assertFalse(mask[2,:,3:].any())
        np.testing.assert_array_equal(x['pages'].reshape(4,2)[1],[9,9])
    def test_restore_truncated_and_reordered_path(self):
        previous=dict(parents=[-1,-1,0,1,3],length=7)
        self.assertEqual(restore_plan(previous,11,5,[7,2,5],4),([2,4,5],[11,20,21]))
        self.assertEqual(restore_plan(previous,9,5,[7,2,5],4),([2],[11]))
        self.assertEqual(restore_plan(previous,8,0,[7,2,5],4),([],[]))
        with self.assertRaises(ValueError): restore_plan(previous,12,5,[7,2,5],4)
    def test_bad_parent_and_unallocated_page(self):
        with self.assertRaises(ValueError):
            build_forward_inputs([dict(tokens=[1,2],parents=[0],prefix=0,blocks=[1])],2,1,1,8)
        with self.assertRaises(ValueError):
            build_forward_inputs([dict(tokens=[1,2],parents=[-1],prefix=7,blocks=[1,-1])],2,1,2,8)

class Sampling(unittest.TestCase):
    def test_proxy_matches_existing_single_tree_per_row(self):
        torch.manual_seed(412)
        pars=[[-1,-1,0,1,3],[-1,0,1],[]]; sibs=[[0,1,0,0,0],[0,0,0],[]]
        B,N,V=3,5,13
        topo=pack_topologies(pars,sibs,N,'cpu')
        p=torch.randn(B,N+1,V);q=torch.randn(B,N,V)
        tok=torch.tensor([[1,3,2,5,6],[2,4,6,0,0],[0,0,0,0,0]])
        # Siblings share the same original parent distribution.
        q[0,1]=q[0,0]
        pos,packed=candidates(p,q,tok,topo,7,5,5)
        for b in range(B):
            ref=tree_proxy_candidates_fixed(p[b],q[b],tok[b],{k:v[b] for k,v in topo.items()},7,5,5)
            self.assertTrue(torch.equal(pos[b],ref[0]))
            self.assertTrue(torch.equal(packed[b],ref[1]))
    def test_greedy_mixed_tree_chain_empty(self):
        pars=[[-1,-1,0,1],[-1,0],[]]; sibs=[[0,1,0,0],[0,0],[]]
        p=torch.zeros(3,5,7)
        p[0,0,2]=p[0,2,4]=p[0,4,6]=9
        p[1,0,3]=p[1,1,5]=p[1,2,1]=9;p[2,0,6]=9
        tok=torch.tensor([[1,2,3,4],[3,5,0,0],[0,0,0,0]])
        out=verify_batch(p,torch.zeros(3,4,7),tok,pack_topologies(pars,sibs,4,'cpu'),[0]*3,[0]*3,greedy=True)
        self.assertEqual(out[0].tolist(),[[1,3,-1,-1],[0,1,-1,-1],[-1]*4])
        self.assertEqual(out[1].tolist(),[[6],[1],[6]])
        self.assertEqual(out[2].tolist(),[[4],[2],[0]])
    def test_exact_ladder_against_independent_scalar_arithmetic(self):
        torch.manual_seed(392)
        pars=[[-1,-1,-1,0,0,2],[-1,0,1]];sibs=[[0,1,2,0,1,0],[0,0,0]]
        B,N,V=2,6,11
        tokens=torch.tensor([[1,4,6,2,8,9],[2,3,4,0,0,0]])
        p=torch.randn(B,N+1,V).softmax(-1);q=torch.randn(B,N,V).softmax(-1)
        q[0,1]=q[0,2]=q[0,0];q[0,4]=q[0,3]
        alpha,_,residual=ladder(tokens,p,q,pack_topologies(pars,sibs,N,'cpu'),N,exact=True)
        for b in range(B):
            for ctx in range(len(pars[b])+1):
                children=[j for j,par in enumerate(pars[b]) if par==ctx-1]
                R=p[b,ctx].clone()
                if children:
                    D=q[b,children[0]].clone()
                    for j in children:
                        t=tokens[b,j];s=sibs[b][j]
                        self.assertAlmostEqual(alpha[b,ctx,s].item(),min(1,(R[t]/D[t]).item()),places=6)
                        R=(R-D).clamp_min(0);R=R/R.sum() if R.sum()>1e-12 else R
                        D[t]=0;D=D/D.sum() if D.sum()>1e-12 else D
                if R.sum()<=1e-12:R=p[b,ctx]
                torch.testing.assert_close(residual[b,ctx],R)
    def test_sampled_siblings_preserve_target_first_token(self):
        # Independent WOR proposal draw, then exact acceptance/recovery. The
        # emitted token must follow target p, even when q is very different.
        torch.manual_seed(144)
        B,N,V=40000,3,5
        p=torch.tensor([.05,.5,.1,.3,.05]);q=torch.tensor([.5,.03,.4,.02,.05])
        tok=torch.multinomial(q.expand(B,V),N,replacement=False)
        base=pack_topologies([[-1]*N],[[0,1,2]],N,'cpu')
        topo={k:v.expand(B,*v.shape[1:]) for k,v in base.items()}
        path,recovery,_=verify_batch(p.log().expand(B,N+1,V),q.log().expand(B,N,V),tok,topo,[1.]*B,[1.]*B)
        emitted=torch.where(path[:,0]>=0,tok.gather(1,path[:,:1].clamp_min(0))[:,0],recovery[:,0])
        empirical=torch.bincount(emitted,minlength=V)/B
        self.assertLess((empirical-p).abs().max().item(),.009)

if __name__=='__main__':unittest.main()

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ForwardAttention(unittest.TestCase):
    @torch.inference_mode()
    def test_captured_ancestor_attention_matches_independent_dense(self):
        from types import SimpleNamespace
        from ssd.layers.attention import Attention
        from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
        dev=torch.device('cuda:0');dtype=torch.float16
        torch.manual_seed(57)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed=torch.nn.Embedding(41,128,device=dev,dtype=dtype)
                self.pos=torch.nn.Embedding(64,128,device=dev,dtype=dtype)
                self.attn=Attention(2,64,64**-.5,2,draft=True,speculate=True,draft_async=True)
                self.attn.k_cache=torch.randn(16,8,2,64,device=dev,dtype=dtype)*.1
                self.attn.v_cache=torch.randn_like(self.attn.k_cache)*.1
            def forward(self,ids,rope):
                x=(self.embed(ids)+self.pos(rope))*.2
                return self.attn(x,x,x)
            def compute_logits(self,x,last_only):return x
        model=Model()
        runner=SimpleNamespace(model=model,device=dev,block_size=8,num_tp_gpus=1,
            config=SimpleNamespace(max_blocks=4),hf_config=SimpleNamespace(torch_dtype=dtype,
                num_attention_heads=2,num_key_value_heads=2,head_dim=64,hidden_size=128))
        rows=[dict(tokens=[1,2,3,4,5],parents=[-1,-1,0,1],prefix=3,blocks=[2,5]),
              dict(tokens=[6,7,8],parents=[-1,0],prefix=9,blocks=[7,4]),
              dict(tokens=[9],parents=[],prefix=0,blocks=[11])]
        forward=BatchedTreeForward(runner,split=False)
        def reference():
            from ssd.engine.helpers.tree_host_topology import context_topology
            outputs=[]
            for row in rows:
                depths,_,visible=context_topology(row['parents']);p=row['prefix']
                ids=torch.tensor(row['tokens'],device=dev);pos=torch.tensor([p]+[p+d+1 for d in depths],device=dev)
                x=((model.embed(ids)+model.pos(pos))*.2).view(-1,2,64)
                slots=torch.tensor([row['blocks'][j//8]*8+j%8 for j in range(p)],device=dev,dtype=torch.long)
                k=torch.cat([model.attn.k_cache.view(-1,2,64)[slots],x]);v=torch.cat([model.attn.v_cache.view(-1,2,64)[slots],x])
                mask=torch.cat([torch.ones(len(ids),p,device=dev,dtype=torch.bool),torch.as_tensor(visible,device=dev).bool()],1)
                scores=torch.einsum('qhd,khd->hqk',x.float(),k.float())*64**-.5
                probs=scores.masked_fill(~mask[None],float('-inf')).softmax(-1)
                outputs.append(torch.einsum('hqk,khd->qhd',probs,v.float()).reshape(len(ids),128))
            return outputs
        expected=reference();out=forward.run(rows,5)
        for b,y in enumerate(expected):torch.testing.assert_close(out[b,:len(y)].float(),y,atol=.001,rtol=.005)
        # Prefix mutation affects only its owner, graph reuses updated values.
        model.attn.v_cache[7,:].add_(.7)
        expected=reference();out2=forward.run(rows,5)
        for b,y in enumerate(expected):torch.testing.assert_close(out2[b,:len(y)].float(),y,atol=.001,rtol=.005)
        torch.testing.assert_close(out[0],out2[0],atol=0,rtol=0)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CapturedAcceptance(unittest.TestCase):
    @torch.inference_mode()
    def test_graph_uses_updated_topology_and_greedy_logits(self):
        from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept
        dev=torch.device('cuda:0');B,N,V=4,5,11
        proxy=BatchedTreeProxy(B,N,V,torch.float16,dev,7,N,5)
        graph=BatchedTreeAccept(proxy,None,1,True)
        torch.manual_seed(9)
        for pars,sibs in [([[-1,-1,0,1],[-1,0],[]],[[0,1,0,0],[0,0],[]]),
                          ([[-1,0,1],[-1,-1,-1]],[[0,0,0],[0,1,2]])]:
            b=len(pars);tok=torch.randint(0,V,(b,N),device=dev)
            p=torch.randn(b,N+1,V,device=dev,dtype=torch.float16);q=torch.randn(b,N,V,device=dev,dtype=torch.float16)
            proxy.prepare(pars,sibs,tok,q)
            ref=verify_batch(p,q,tok,{k:v[:b] for k,v in proxy.topology.items()},[0]*b,[0]*b,greedy=True)
            out=graph.replay(p,[0]*b,[0]*b)
            for a,z in zip(out,ref):torch.testing.assert_close(a,z)
    @torch.inference_mode()
    def test_stochastic_graph_refreshes_rng_and_recovers_target(self):
        from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept
        dev=torch.device('cuda:0');B,N,V=128,1,5
        proxy=BatchedTreeProxy(B,N,V,torch.float32,dev,3,N,3)
        graph=BatchedTreeAccept(proxy,None,1,False)
        # Empty proposals reduce exactly to direct target sampling. This also
        # verifies RNG advances on replay instead of repeating capture draws.
        p=torch.tensor([.03,.51,.09,.3,.07],device=dev)
        logits=p.log().expand(B,N+1,V)
        counts=torch.zeros(V,device=dev);first=None;changed=False
        for _ in range(200):
            out=graph.replay(logits,[1]*B,[1]*B)[1][:,0]
            if first is None:first=out.clone()
            else:changed|=not torch.equal(first,out)
            counts+=torch.bincount(out,minlength=V)
        self.assertTrue(changed)
        self.assertLess((counts/counts.sum()-p).abs().max().item(),.015)

class RootRanking(unittest.TestCase):
    def test_batched_p1_matches_ragged_single_request_policy(self):
        from ssd.engine.helpers.batch_tree_roots import root_candidates
        from ssd.engine.helpers.p1_tree import build_uniform_p1_roots
        from ssd.engine.helpers.tree_host_topology import context_topology
        torch.manual_seed(413)
        B,P,V,U=3,6,31,2
        logits=torch.randn(B,P,V)
        tokens=torch.randint(V,(B,P))
        parents=[[-1,-1,0,1],[-1,0],[]]
        lens=torch.tensor([len(p)+1 for p in parents]);temps=torch.tensor([.7,0.,1.2])
        vis=torch.zeros(B,P,P,dtype=torch.uint8)
        for b,par in enumerate(parents):vis[b,:lens[b],:lens[b]]=torch.as_tensor(context_topology(par)[2])
        for sampler_x in (None,1.3):
            tok,score=root_candidates(logits,tokens,vis,lens,temps,U,sampler_x,3)
            for b,n in enumerate(lens.tolist()):
                ref=build_uniform_p1_roots(logits[b,:n],tokens[b,:n],U,temps[b].expand(n),
                    sampler_x=sampler_x,async_fan_out=3,context_glue_rows=vis[b,:n,:n])
                torch.testing.assert_close(tok[b,:n*U],ref['tokens'])
                torch.testing.assert_close(score[b,:n*U],ref['scores'],atol=1e-7,rtol=2e-6)
                self.assertFalse(score[b,n*U:].any())
