"""Independent formula, allocation and serving-contract checks for local research."""
import os
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ssd.engine.helpers import root_policy as rp
from ssd.engine.helpers import batch_tree_sampling as dense
from ssd.engine.helpers import flat_tree_sampling as flat
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.helpers.tree_expansion_policy import TreeExpansionPolicy
from ssd.engine.helpers.tree_gain_allocation import make_constants,allocate,exhaustive


def scalar_tree(e,q,tokens,parents,siblings,mix):
    """Python/FP64 enumeration of the two terminal recursions, independent of GPU layout."""
    n=len(parents);children=[[] for _ in range(n+1)]
    for j,(p,s) in enumerate(zip(parents,siblings)):children[p+1].append((s,j))
    reach=[1.]+[0.]*n;average=[1.]+[0.]*n;term=[]
    for ctx,cs in enumerate(children):
        cs.sort();r=e[ctx].copy();d=q[cs[0][1]].copy() if cs else None
        observed=expected=1.
        for _,j in cs:
            y=tokens[j];a=min(1.,r[y]/(d[y]+1e-10));av=np.minimum(r,d).sum()
            reach[j+1]=reach[ctx]*observed*a
            average[j+1]=average[ctx]*expected*av
            observed*=1-a;expected*=1-av
            nr=np.maximum(r-d,0);r=nr/nr.sum() if nr.sum()>1e-12 else np.zeros_like(nr)
            d[y]=0;d=d/d.sum() if d.sum()>1e-12 else np.zeros_like(d)
        term.append((1-mix)*reach[ctx]*observed+mix*average[ctx]*expected)
    return np.asarray(term)


class LocalRootMath(unittest.TestCase):
    def test_scalar_chain_scores_and_full_mass(self):
        e=torch.tensor([[[.8,.1,.06,.04],[.1,.6,.2,.1],[.3,.4,.2,.1]]]).log()
        q=torch.tensor([[[.5,.3,.1,.1],[.3,.4,.2,.1]]]).log();tok=torch.tensor([[0,1]])
        for temperature in (0.,.5,.7,1.):
            pe=(e.double()/max(temperature,1e-20) if temperature else e.double()).softmax(-1)[0].numpy()
            pq=(q.double()/max(temperature,1e-20) if temperature else q.double()).softmax(-1)[0].numpy()
            term=scalar_tree(pe,pq,tok[0].tolist(),[-1,0],[0,0],.25)
            expect=[]
            for i in range(3):
                score=pe[i]*(1-pq[i]) if i<2 else pe[i].copy()
                if i<2:score[tok[0,i]]=0
                score=score/score.sum()*term[i]
                expect.extend((float(s),i,v) for v,s in enumerate(score))
            expect=sorted(expect,reverse=True)[:5]
            pos,ids,values=rp.chain_candidates(e,q,tok,torch.tensor([2]),4,5,
                                               target_temps=temperature,draft_temps=temperature)
            self.assertEqual(list(zip(pos[0].tolist(),ids[0].tolist())),[(i,v) for _,i,v in expect])
            torch.testing.assert_close(values[0].double(),torch.tensor([s for s,_,_ in expect],dtype=torch.double),rtol=2e-6,atol=1e-7)

    def test_zero_length_chain_is_a_bonus_distribution(self):
        e=torch.tensor([[[.7,.2,.1]]]).log()
        p,i,v=rp.chain_candidates(e,torch.empty(1,0,3),torch.empty(1,0,dtype=torch.long),torch.tensor([0]),3,3)
        self.assertEqual(p.tolist(),[[0,0,0]])
        self.assertEqual(i.tolist(),[[0,1,2]])
        torch.testing.assert_close(v,e[:,0].exp())

    def test_branching_terminal_mass_and_scalar_reference(self):
        torch.manual_seed(510)
        ps=[-1,-1,0,0,1,2];ss=[0,1,0,1,0,0]
        e=torch.randn(1,7,11).softmax(-1);q=torch.randn(1,6,11).softmax(-1)
        q[:,1]=q[:,0];q[:,3]=q[:,2];tok=torch.tensor([[0,1,2,3,4,5]])
        topo=dense.pack_topologies([ps],[ss],6,'cpu')
        for mix in (0.,.25,1.):
            _,term,_=dense.ladder(tok,e,q,topo,6,overlap_mix=mix)
            ref=scalar_tree(e[0].double().numpy(),q[0].double().numpy(),tok[0].tolist(),ps,ss,mix)
            torch.testing.assert_close(term[0].double(),torch.from_numpy(ref),rtol=3e-6,atol=1e-7)
            self.assertAlmostEqual(term.sum().item(),1.,places=6)

    def test_ragged_and_no_jit_miss_rank_real_bonus(self):
        e=torch.tensor([[[.5,.3,.2],[.8,.15,.05],[.1,.2,.7]]]).log()
        q=torch.tensor([[[.5,.3,.2],[.05,.15,.8]]]).log();tok=torch.tensor([[0,0]])
        pos,ids,val=rp.chain_candidates(e,q,tok,torch.tensor([1]),3,3)
        self.assertEqual(pos.tolist(),[[1,1,1]]);self.assertEqual(ids.tolist(),[[0,1,2]])
        torch.testing.assert_close(val,torch.tensor([[.8,.15,.05]]))
        pos,ids,val=rp.chain_candidates(e,q,tok,torch.tensor([2]),3,3,cache_hits=torch.tensor([False]))
        self.assertEqual(pos.tolist(),[[0,0,0]]);torch.testing.assert_close(val,e[:,0].exp())
        # A no-JIT miss has no meaningful q; even poisoned unused logits
        # must not leak NaNs into the current-context proxy ranking.
        pos,ids,val=rp.chain_candidates(e,torch.full_like(q,float('nan')),tok,
            torch.tensor([2]),3,3,cache_hits=torch.tensor([False]))
        self.assertEqual(pos.tolist(),[[0,0,0]]);torch.testing.assert_close(val,e[:,0].exp())

    def test_chain_tree_flat_match_live_temperature_and_sampler_x(self):
        torch.manual_seed(519);b,n,v=3,4,41
        e=torch.randn(b,n+1,v);q=torch.randn(b,n,v);tok=torch.randint(v,(b,n))
        lengths=[4,2,0];ps=[list(range(-1,z-1)) for z in lengths];ss=[[0]*z for z in lengths]
        topo=dense.pack_topologies(ps,ss,n,'cpu');ft=flat.topology(ps,ss,n,3,'cpu',12)
        temps=torch.tensor([.7,0.,.3]);dt=torch.tensor([.8,0.,.5])
        for sx in (None,.7):
            opts=dict(sampler_x=sx,fan_out=3)
            a=rp.chain_candidates(e,q,tok,torch.tensor(lengths),8,8,True,temps,dt,**opts)
            z=rp.tree_candidates(e,q,tok,topo,8,n,8,temps,dt,**opts)
            f=flat.candidates(e[ft['req'],ft['local']],flat.pack_nodes(q,ft),flat.pack_nodes(tok,ft),ft,8,n,8,temps,dt,**opts)
            for x,y in zip(a[:2],z[:2]):torch.testing.assert_close(x,y)
            for x,y in zip(a[:2],f):torch.testing.assert_close(x,y)

    def test_proxy_rank_has_no_rng_or_input_mutation(self):
        torch.manual_seed(41);e=torch.randn(2,5,31);q=torch.randn(2,4,31);tok=torch.randint(31,(2,4))
        saved=[x.clone() for x in (e,q,tok)];rng=torch.get_rng_state().clone()
        rp.chain_candidates(e,q,tok,torch.tensor([4,2]),5,9)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for a,z in zip((e,q,tok),saved):self.assertTrue(torch.equal(a,z))

    def test_full_normalization_does_not_inflate_diffuse_position(self):
        scores=torch.tensor([[[.4,.3,.3],[.9,.06,.04]]]);h=torch.tensor([[.6,.4]])
        old=rp.rank(scores,h,1,1,'topm',False);new=rp.rank(scores,h,1,1,'full',False)
        self.assertEqual(old[0].item(),0);self.assertEqual(new[0].item(),1)
        self.assertAlmostEqual(new[2].item(),.36,places=6)


class LocalTreeMath(unittest.TestCase):
    def test_allocation_matches_exhaustive_including_nonconcave_gain(self):
        rng=np.random.default_rng(56)
        for curve in ([0.,.1,.9,.95],[0.,.6,.8,.9]):
            c=make_constants(6,curve)
            for _ in range(30):
                weights=rng.uniform(.01,.99,(2,3));root=torch.tensor([0,0,0,1,1,1])
                remaining=torch.tensor(rng.integers(1,7,size=2));future=int(rng.integers(0,3))
                fan=allocate(root,torch.tensor(np.log(weights).ravel()),torch.ones(6,dtype=torch.bool),remaining,future,c)
                for k in range(2):
                    budget=max(1,int(remaining[k])-future);best,_=exhaustive(weights[k],curve,budget)
                    got=sum(w*curve[x] for w,x in zip(weights[k],fan[k*3:k*3+3]))
                    self.assertAlmostEqual(got,best,places=12)
                changed=torch.tensor(np.log(weights).ravel());changed[3:]+=10
                other=allocate(root,changed,torch.ones(6,dtype=torch.bool),remaining,future,c)
                self.assertTrue(torch.equal(fan[:3],other[:3]))

    def test_reach_siblings_and_preserve_true_q(self):
        p=TreeExpansionPolicy.__new__(TreeExpansionPolicy)
        p.alpha=torch.tensor([[.8],[.5],[.25]],dtype=torch.double);p.bins=torch.empty(0,dtype=torch.double)
        raw=torch.tensor([[.6,.3,.1]],dtype=torch.double);before=raw.clone()
        got=p.priority(torch.tensor([.5],dtype=torch.double).log(),torch.tensor([0,0,0]),raw)
        torch.testing.assert_close(got.exp(),torch.tensor([.4,.05,.0125],dtype=torch.double))
        self.assertTrue(torch.equal(raw,before))

    def test_frontier_only_reconsiders_unexpanded_nodes(self):
        ar=pt.TreeArena(8,'cpu');ar.n=5
        ar.valid[:5]=True;ar.state[:5]=torch.tensor([1,0,1,0,0]);ar.depth[:5]=torch.tensor([0,1,1,2,2]);ar.root[:5]=0
        ar.logpri[:5]=torch.tensor([0.,-.1,-.2,-1.,-2.]);ar.raw_q[:5]=.5
        old=pt._arena_select_global(ar,1,2,4,torch.tensor([4]),1,1)[0]
        new=pt._arena_select_global(ar,1,2,4,torch.tensor([4]),1,1,all_frontier=True)[0]
        self.assertEqual(old.item(),3);self.assertEqual(new.item(),1)

    def test_packaged_calibration_g_gt_m_and_explicit_limits(self):
        for c,nv,m in ((1,4,4),(3,14,12),(3,18,12),(4,4,4)):
            ex=SimpleNamespace(C=c,NV=nv,phase='p1',policy='dynamic',dev='cpu',
                cfg=SimpleNamespace(duet_p1_tree_policy='on',duet_p1_tree_verify_nodes=m))
            with patch.dict(os.environ,SSD_TREE_EXPANSION_POLICY='reach_gain_frontier'):
                p=TreeExpansionPolicy(ex)
            self.assertEqual(p.mode,'q_path' if c==4 else 'reach_gain_frontier')
            if c<4:self.assertIsNotNone(p.gain)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA')
class LocalGraphs(unittest.TestCase):
    @torch.inference_mode()
    def test_graphs_use_live_temps_and_actual_q(self):
        torch.manual_seed(516);dev='cuda';b,n,v=2,4,47
        e=torch.randn(b,n+1,v,device=dev);q=torch.randn(b,n,v,device=dev);tok=torch.randint(v,(b,n),device=dev)
        policy=dict(sampler_x=.7,fan_out=3)
        dg=dense.BatchedTreeProxy(b,n,v,torch.float32,dev,8,n,8,policy=policy)
        fg=flat.FlatProxy(b,n,v,torch.float32,dev,8,n,8,policy=policy)
        cg=pt.ChainProxyCUDAGraph(n,v,8,8,True,torch.float32,dev,policy=policy)
        for temps in ([.7,.3],[0.,1.],[1.,.7]):
            ps=[[-1,0,1,2],[-1,0]];ss=[[0]*4,[0]*2]
            dg.prepare(ps,ss,tok,q,temps,temps);fg.prepare(ps,ss,tok,q,temps,temps)
            top=fg.current['topo'];a=dg.replay(e);z=fg.replay(e[top['req'],top['local']])
            for x,y in zip(a,z):self.assertTrue(torch.equal(x,y))
            c=cg.replay(e[0],q[0],tok[0],temps[0],temps[0])
            for x,y in zip(a,c[:2]):self.assertTrue(torch.equal(x[0],y))

    @torch.inference_mode()
    @patch.dict(os.environ, {'SSD_TREE_EXPANSION_POLICY':'reach_gain_frontier'})
    def test_research_executor_eager_graph_and_proposal_q(self):
        if __package__:
            from . import test_p2_executor_parity as fixture
        else:
            import test_p2_executor_parity as fixture
        from ssd.engine.helpers.p1_tree import P1TreeExecutor
        f=fixture.TestExecutorModuleParity()
        for phase in ('p1','p2'):
            with patch.object(fixture._MiniCfg,'duet_p1_tree_policy','on',create=True), \
                 patch.object(fixture._MiniCfg,'duet_p2_tree_policy','on',create=True):
                ex,cfg,page,v=f._mk('cuda',policy='dynamic')
                if phase=='p1':
                    ex=P1TreeExecutor(ex.model,ex.model.logits_fn,cfg,'cuda',page,8,
                        v,4,2,64,context_bucket=3,materialize_backbone_logits=False)
            self.assertEqual(ex.expansion_policy.mode,'reach_gain_frontier')
            ex.debug_buffers_enabled=True
            ex.prepare_bucket(2);f._fill_inputs(ex,page,page+21)
            g=torch.Generator().manual_seed(233)
            ex.parity_noise=[torch.empty(w,v).exponential_(1,generator=g).cuda()
                             for w in ex.round_widths]
            ex._local_idx=torch.full((ex.arena.capacity,),-1,dtype=torch.long,device='cuda')
            ex.model.cache.zero_();ex.run_once(2)
            keys=('view_tok','view_par','view_sib','view_rawq','view_pcell','out_valid')
            expected={k:getattr(ex,k).clone() for k in keys}
            ex.model.cache.zero_();ex.capture(2);ex.model.cache.zero_();ex.replay(2)
            for k,a in expected.items():
                torch.testing.assert_close(a,getattr(ex,k),rtol=1e-5,atol=1e-6)
            for root in range(ex.R):
                n=int(ex.out_valid[root])
                self.assertLessEqual(n,ex.NV)
                if n:
                    cells=ex.view_pcell[root,:n]
                    probs=pt.q_probs_from_logits(ex.cell_logits[cells],ex.in_temps[:1].expand(n))
                    raw=probs.gather(1,ex.view_tok[root,:n,None])[:,0]
                    torch.testing.assert_close(raw,ex.view_rawq[root,:n].float(),rtol=1e-5,atol=1e-6)

if __name__=='__main__':unittest.main()
