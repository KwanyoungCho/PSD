"""Score selection and target-law preservation, including selection bias."""
import itertools
import math
import unittest
from types import SimpleNamespace as NS
import torch
from ssd.engine.helpers.score_tree import select_tree,generation_capacity
from ssd.engine.helpers.target_tree_sampling import verify_dense,verify_flat
from ssd.engine.helpers.batch_tree_sampling import pack_topologies
from ssd.engine.helpers.flat_tree_sampling import topology,pack_nodes


class ScoreSelection(unittest.TestCase):
    def test_later_sibling_selected_and_renumbered(self):
        self.assertEqual(select_tree([-1,-1],[.1,.8],1),([1],[-1],[0]))

    def test_high_probability_branch_and_ancestors(self):
        self.assertEqual(select_tree([-1,-1,0,1],[.01,.99,.01,.99],3),
            ([0,1,3],[-1,-1,1],[0,1,0]))

    def test_zero_probability_and_unit_probability_ties(self):
        self.assertEqual(select_tree([-1,0,1],[1.,1.,1.],2)[0],[0,1])
        self.assertEqual(select_tree([-1,0,1],[0.,0.,0.],2)[0],[0,1])

    def test_nonbinding_and_invalid(self):
        self.assertEqual(select_tree([-1,0],[.8,.9],20)[0],[0,1])
        for p,q in [([1],[.2]),([-1],[1.1]),([-1],[float('nan')])]:
            with self.assertRaises(ValueError):select_tree(p,q,1)

    def test_natural_generation_bound(self):
        self.assertEqual(generation_capacity((96,)+(56,)*9,3),1800)
        self.assertEqual(generation_capacity((28,)*5,3),420)

    def test_single_and_bulk_export_same_score_policy(self):
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        def service():
            s=object.__new__(BatchedDuetDraft);s.entries={}
            s.cfg=NS(duet_p2_tree_policy='on',duet_p2_tree_verify_nodes=3)
            return s
        ar=NS(NV=4,out_valid=torch.tensor([4]),view_tok=torch.tensor([[8,2,6,5]]),
            view_par=torch.tensor([[-1,-1,0,1]]),view_sib=torch.tensor([[0,1,0,0]]),
            view_pcell=torch.tensor([[3,3,7,8]]),view_rawq=torch.tensor([[.01,.99,.01,.99]]),
            cell_logits=torch.randn(9,11))
        a=service();b=service();args=(ar,31,torch.tensor([1]),torch.tensor([9]),2,True)
        a._save_entries(*args);b._save_entries_bulk([args])
        for s in (a,b):
            e=s.entries[31,1,9]
            self.assertEqual(e['tokens'],[8,2,5]);self.assertEqual(e['parents'],[-1,-1,1])
            self.assertEqual(e['siblings'],[0,1,0]);self.assertEqual(e['cells'],[3,3,8])

    def test_b1_and_b4_share_safe_tree_route(self):
        from ssd.engine.helpers.batch_tree_common import enabled
        for b in (1,4):self.assertTrue(enabled(NS(duet_enabled=True,duet_tree_enabled=True,max_num_seqs=b)))
        self.assertFalse(enabled(NS(duet_enabled=True,duet_tree_enabled=False,max_num_seqs=1)))

    def test_phase_caps_have_separate_forward_shapes(self):
        from ssd.engine.helpers.batch_tree_common import verification_widths
        cfg=NS(duet_phase1_k=10,duet_phase2_k=5,duet_response_token_width=23,
               duet_p1_tree_policy='on',duet_p2_tree_policy='on',
               duet_p1_tree_verify_nodes=23,duet_p2_tree_verify_nodes=15)
        self.assertEqual(verification_widths(cfg),(5,10,15,23))

    def test_actual_arena_export_matches_independent_root_local_views(self):
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        from ssd.engine.helpers.p2_tree import TreeArena
        a=TreeArena(8,'cpu',max_cells=4)
        a.valid[:6]=True;a.n.fill_(6)
        a.tok[:6]=torch.tensor([7,9,8,2,6,5])
        a.root[:6]=torch.tensor([0,1,0,0,0,0])
        a.parent_idx[:6]=torch.tensor([-1,-1,0,0,2,3])
        a.sib[:6]=torch.tensor([0,0,0,1,0,0])
        a.raw_q[:6]=torch.tensor([1.,1.,.01,.99,.01,.99],dtype=torch.float64)
        a.depth[:6]=torch.tensor([0,0,1,1,2,2]);a.cell[:6]=torch.tensor([3,-1,7,8,-1,-1])
        ex=NS(arena=a,R=2,cell_logits=torch.randn(9,11))
        s=object.__new__(BatchedDuetDraft);s.entries={}
        s.cfg=NS(duet_p2_tree_policy='on',duet_p2_tree_verify_nodes=3)
        s._save_entries_bulk([(ex,31,torch.tensor([1,2]),torch.tensor([7,9]),2,True)])
        e=s.entries[31,1,7]
        self.assertEqual(e['tokens'],[8,2,5]);self.assertEqual(e['parents'],[-1,-1,1])
        self.assertEqual(e['siblings'],[0,1,0]);self.assertEqual(e['cells'],[3,3,8])
        self.assertNotIn((31,2,9),s.entries)


class TargetLaw(unittest.TestCase):
    def test_q_equal_p_does_not_save_score_pruned_original_q_verifier(self):
        from ssd.engine.helpers.p2_tree import rerank_tree_indices
        q=[.6,.3,.1];out=[0.,0.,0.]
        for a,b,c,d in itertools.product(range(3),repeat=4):
            if a==b or c==d:continue
            w=q[a]*q[b]/(1-q[a])*q[c]*q[d]/(1-q[c])
            keep=rerank_tree_indices([-1,-1,0,0],[0,1,0,1],[q[a],q[b],q[c],q[d]],2)
            if 2 in keep:out[c]+=w
            else:
                for j in range(3):out[j]+=w*q[j]
        self.assertAlmostEqual(sum(out),1.)
        self.assertAlmostEqual(out[0],.7272857142857143)
        self.assertGreater(.5*sum(abs(x-y) for x,y in zip(out,q)),.12)

    def test_exact_two_token_target_law_for_arbitrary_pruned_trees(self):
        p=torch.tensor([.6,.3,.1],dtype=torch.float64)
        transition=torch.tensor([[.2,.7,.1],[.3,.2,.5],[.8,.1,.1]],dtype=torch.float64)
        for tokens,par,sib in [([0,1,2,0],[-1,-1,0,1],[0,1,0,0]),
                               ([1,2],[-1,0],[0,0]),([2],[-1],[0]),([],[],[])]:
            n=len(tokens);r=n+1
            distributions=torch.cat([p[None],transition[tokens]],0)
            draws=torch.tensor(list(itertools.product(range(3),repeat=r)))
            weights=distributions[torch.arange(r)[None],draws].prod(1)
            b=len(draws);top=pack_topologies([par],[sib],n,'cpu')
            top={k:v.expand(b,*v.shape[1:]) for k,v in top.items()}
            tok=torch.tensor(tokens,dtype=torch.long).expand(b,n)
            paths,recovery,_=verify_dense(distributions.log().expand(b,r,3),tok,top,[1.]*b,draws=draws)
            joint=torch.zeros(3,3,dtype=torch.float64)
            for i,w in enumerate(weights):
                first=int(draws[i,0])
                if n and paths[i,0]>=0:
                    node=int(paths[i,0]);second=int(draws[i,node+1]);joint[first,second]+=w
                    emitted_second=int(tok[i,paths[i,1]]) if n>1 and paths[i,1]>=0 else int(recovery[i,0])
                    self.assertEqual(emitted_second,second)
                else:joint[first]+=w*transition[first]
            torch.testing.assert_close(joint,p[:,None]*transition,rtol=1e-12,atol=1e-12)

    def test_flat_dense_match_on_same_context_draws(self):
        pars=[[-1,-1,0,1],[-1,0],[]];sibs=[[0,1,0,0],[0,0],[]]
        tokens=torch.tensor([[1,2,3,4],[3,5,0,0],[0,0,0,0]])
        log=torch.randn(3,5,7);draws=torch.tensor([[2,1,4,2,6],[3,5,1,0,0],[6,0,0,0,0]])
        dense=verify_dense(log,tokens,pack_topologies(pars,sibs,4,'cpu'),[.7]*3,draws=draws)
        topo=topology(pars,sibs,4,3,'cpu')
        flat=verify_flat(log[topo['req'],topo['local']],pack_nodes(tokens,topo),topo,4,[.7]*3,
            draws=draws[topo['req'],topo['local']])
        for x,y in zip(dense,flat):self.assertTrue(torch.equal(x,y))

    def test_greedy_matches_target_argmax_without_rng(self):
        torch.manual_seed(173);log=torch.randn(2,5,7)
        token=torch.tensor([[1,2,3,4],[2,3,0,0]])
        topo=pack_topologies([[-1,-1,0,1],[-1,0]],[[0,1,0,0],[0,0]],4,'cpu')
        before=torch.get_rng_state()
        out=verify_dense(log,token,topo,[0.,0.],greedy=True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        ref=verify_dense(log,token,topo,[0.,0.],draws=log.argmax(-1))
        for x,y in zip(out,ref):self.assertTrue(torch.equal(x,y))

    def test_stochastic_empirical_target_law(self):
        b=40000;n=2;p=torch.tensor([.1,.6,.3]);tokens=torch.tensor([[0,2]]).expand(b,n)
        topo=pack_topologies([[-1,-1]],[[0,1]],n,'cpu')
        topo={k:v.expand(b,*v.shape[1:]) for k,v in topo.items()}
        torch.manual_seed(445)
        paths,recovery,_=verify_dense(p.log().expand(b,n+1,3),tokens,topo,[1.]*b,stochastic_only=True)
        value=torch.where(paths[:,0]>=0,tokens.gather(1,paths[:,:1].clamp_min(0))[:,0],recovery[:,0])
        observed=torch.bincount(value,minlength=3)/b
        self.assertLess((observed-p).abs().max().item(),.009)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class Captured(unittest.TestCase):
    def test_score_export_matches_cpu_reference_and_replays_updated_q(self):
        from ssd.engine.helpers.p2_tree import TreeArena
        from ssd.engine.helpers.score_tree_export import ScoreTreeExport
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        dev='cuda:0';a=TreeArena(64,dev,max_cells=32)
        parents=[-1,-1,-1,0,0,1,1,2,3,4,5,6,7,8,9,10]
        roots=[0,1,2,0,0,1,1,2,0,0,1,1,2,0,0,1]
        depths=[0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3]
        n=len(parents)
        a.valid[:n]=True;a.n.fill_(n)
        a.tok[:n]=torch.arange(n,device=dev)+1
        a.parent_idx[:n]=torch.tensor(parents,device=dev)
        a.root[:n]=torch.tensor(roots,device=dev);a.depth[:n]=torch.tensor(depths,device=dev)
        a.cell[:n]=torch.arange(n,device=dev)
        ex=NS(arena=a,R=3,F=3,cell_logits=torch.randn(32,7,device=dev))
        contexts=torch.tensor([0,1,2],device=dev);tokens=torch.tensor([1,2,3],device=dev)
        def service():
            s=object.__new__(BatchedDuetDraft);s.entries={}
            s.cfg=NS(duet_p2_tree_policy='on',duet_p2_tree_verify_nodes=3);return s
        selector=ScoreTreeExport(ex,3)
        junk=[torch.full((3,),97+i,device=dev,dtype=torch.long) for i in range(10)]
        for q in ([1.]*n,[1.,1.,1.,.01,.9,.8,.3,.7,.01,.99,.2,.8,.4,.9,.9,.9],
                  [1.,1.,1.]+[0.]*(n-3)):
            a.raw_q[:n]=torch.tensor(q,dtype=torch.float64,device=dev)
            before=torch.cuda.get_rng_state(dev)
            reference=service();actual=service()
            reference._save_entries_bulk([(ex,11,contexts,tokens,2,True)])
            ex.score_export=selector
            actual._save_entries_bulk([(ex,11,contexts,tokens,2,True)])
            del ex.score_export
            self.assertTrue(torch.equal(before,torch.cuda.get_rng_state(dev)))
            self.assertEqual(set(reference.entries),set(actual.entries))
            for key,e in reference.entries.items():
                for field in ('tokens','parents','siblings','cells','generated_nodes'):
                    self.assertEqual(e[field],actual.entries[key][field],(q,key,field))

    def test_dense_and_flat_graph_greedy_and_stochastic_b1_b4(self):
        from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept
        from ssd.engine.helpers.flat_tree_sampling import FlatProxy,FlatAccept
        for b in (1,4):
            for compact in (False,True):
                for greedy in (False,True):
                    with self.subTest(batch=b,flat=compact,greedy=greedy):
                        n=8;v=11;dev='cuda:0';dtype=torch.bfloat16
                        proxy_cls=FlatProxy if compact else BatchedTreeProxy
                        accept_cls=FlatAccept if compact else BatchedTreeAccept
                        proxy=proxy_cls(b,n,v,dtype,dev,8,n,4,c_max=3)
                        parents=[[-1,0]]*b;siblings=[[0,0]]*b
                        tokens=torch.tensor([[1,2]+[0]*(n-2)]*b,device=dev)
                        proxy.prepare(parents,siblings,tokens,torch.randn(b,n,v,device=dev),[0. if greedy else .7]*b,[.7]*b)
                        graph=accept_cls(proxy,None,1,greedy,stochastic_only=not greedy,target_only=True)
                        logits=torch.zeros(b,n+1,v,dtype=dtype,device=dev)
                        logits[:,0,1]=logits[:,1,2]=logits[:,2,3]=8
                        inp=logits[proxy.current['topo']['req'],proxy.current['topo']['local']] if compact else logits
                        out=graph.replay(inp,[0. if greedy else .7]*b,[.7]*b)
                        torch.cuda.synchronize()
                        self.assertEqual(out[0].shape,(b,n))
                        if greedy:
                            self.assertEqual(out[0].tolist(),[[0,1]+[-1]*(n-2)]*b)
                            self.assertEqual(out[1].tolist(),[[3]]*b)


if __name__=='__main__':unittest.main()
