"""Native root settings: independent scores, serving parity and graph safety."""
import os
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
import unittest
from unittest.mock import patch
import torch
from test_blackwell_review import make_config
from ssd.engine.helpers import root_policy as rp
from ssd.engine.helpers import batch_tree_sampling as dense
from ssd.engine.helpers import flat_tree_sampling as flat
from test_local_research_policy import scalar_tree


class RootSettings(unittest.TestCase):
    def test_topm_default_is_consistent_and_full_remains_explicit(self):
        import ast
        import inspect
        from pathlib import Path
        from ssd.presets import duet_70b_paper_kwargs
        cfg,draft=make_config()
        for c in (cfg,draft,None):
            self.assertEqual(rp.options(c)['normalization'],'topm')
        self.assertEqual(duet_70b_paper_kwargs('/draft')['duet_root_normalization'],'topm')
        self.assertEqual(duet_70b_paper_kwargs('/draft',normalization='full')['duet_root_normalization'],'full')
        for func in (rp.chain_candidates,rp.tree_candidates,flat.candidates):
            self.assertEqual(inspect.signature(func).parameters['normalization'].default,'topm')
        # Inspect argparse's literal without importing benchmark-only dependencies.
        path=Path(__file__).resolve().parents[1]/'bench/bench.py'
        calls=[n for n in ast.walk(ast.parse(path.read_text()))
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr=='add_argument' and n.args
               and isinstance(n.args[0],ast.Constant)
               and n.args[0].value=='--duet_root_normalization']
        self.assertEqual(len(calls),1)
        self.assertEqual(next(ast.literal_eval(k.value) for k in calls[0].keywords if k.arg=='default'),'topm')

    def test_explicit_dtype_survives_both_config_roles(self):
        for dtype in ('bfloat16','float16'):
            cfg,draft=make_config(dtype=dtype)
            for c in (cfg,draft):
                self.assertEqual(c.hf_config.torch_dtype,getattr(torch,dtype))
                self.assertEqual(c.draft_hf_config.torch_dtype,getattr(torch,dtype))
                self.assertEqual(c.target_hf_config.torch_dtype,getattr(torch,dtype))
        with self.assertRaisesRegex(ValueError,'dtype'):make_config(dtype='float32')

    def test_paper_preset_keeps_dual_tree_and_explicit_budgets(self):
        from ssd.presets import duet_70b_paper_kwargs
        opts=duet_70b_paper_kwargs('/draft',normalization='topm',expansion='frontier')
        cfg,draft=make_config(**opts)
        for c in (cfg,draft):
            self.assertEqual((c.duet_phase1_k,c.duet_phase2_k),(8,4))
            self.assertEqual((c.duet_p1_tree_policy,c.duet_p2_tree_policy),('on','on'))
            self.assertEqual(c.duet_tree_expansion,'frontier')
            self.assertEqual(c.duet_root_normalization,'topm')
            self.assertEqual(c.duet_tree_c_tensor,3)
            self.assertEqual(c.duet_tree_conf_threshold,0)
            self.assertEqual(c.max_num_seqs,1)
            self.assertEqual(c.async_fan_out,3)
            self.assertEqual(c.duet_exit_layer,56)

    def test_config_reconstruction_and_validation(self):
        for source in ('complement', 'residual', 'proxy'):
            for norm in ('topm', 'full'):
                cfg, draft = make_config(duet_root_source=source,
                    duet_root_normalization=norm, duet_root_overlap_mix=.75)
                for c in (cfg, draft):
                    self.assertEqual(rp.options(c), dict(source=source,
                        normalization=norm, overlap_mix=.75, sampler_x=None, fan_out=3))
        for kwargs in (dict(duet_root_source='bad'), dict(duet_root_normalization='bad'),
                       *[dict(duet_root_overlap_mix=x) for x in (-.1, 1.1, float('nan'), float('inf'), True, '0.25')]):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_config(**kwargs)

    def test_independent_temperature_matched_topm_and_full(self):
        torch.manual_seed(193)
        e=torch.randn(1,4,37);q=torch.randn(1,3,37);tok=torch.tensor([[3,9,7]])
        for temp in (.1,.7,1.):
            pe=(e.double()/temp).softmax(-1)[0].numpy()
            pq=(q.double()/temp).softmax(-1)[0].numpy()
            for mix in (0.,.25,1.):
                term=scalar_tree(pe,pq,tok[0].tolist(),[-1,0,1],[0,0,0],mix)
                for source in ('complement','residual','proxy'):
                    for norm in ('topm','full'):
                        expected=[]
                        for i in range(4):
                            score=pe[i].copy()
                            if i<3:
                                if source=='complement': score*=1-pq[i]
                                if source=='residual': score=(score-pq[i]).clip(0)
                                score[tok[0,i]]=0
                            selected=score.argsort()[-5:]
                            denom=score.sum() if norm=='full' else score[selected].sum()
                            expected.extend((float(score[v]/max(denom,1e-10)*term[i]),i,int(v)) for v in selected)
                        expected=sorted(expected,reverse=True)[:7]
                        pos,ids,values=rp.chain_candidates(e,q,tok,torch.tensor([3]),5,7,
                            target_temps=temp,draft_temps=temp,source=source,
                            normalization=norm,overlap_mix=mix)
                        # Zero-score ordering is not part of the ranking contract.
                        actual={(int(p),int(v)):float(s) for p,v,s in zip(pos[0],ids[0],values[0]) if s>1e-8}
                        ref={(p,v):s for s,p,v in expected if s>1e-8}
                        self.assertEqual(set(actual),set(ref))
                        for key in ref:self.assertAlmostEqual(actual[key],ref[key],delta=2e-6)

    def test_branching_dense_flat_both_normalizations(self):
        torch.manual_seed(87);b,n,v=2,4,73
        ps=[[-1,-1,0,1],[-1,0]];ss=[[0,1,0,0],[0,0]]
        topo=dense.pack_topologies(ps,ss,n,'cpu');ft=flat.topology(ps,ss,n,3,'cpu',9)
        e=torch.randn(b,n+1,v);q=torch.randn(b,n,v);q[0,1]=q[0,0]
        tok=torch.randint(v,(b,n));tok[0,1]=(tok[0,0]+1)%v
        for temp in (.1,.7,1.):
            t=torch.full((b,),temp)
            for source in ('complement','residual','proxy'):
                for norm in ('topm','full'):
                    opts=dict(source=source,normalization=norm)
                    a=rp.tree_candidates(e,q,tok,topo,9,n,5,t,t,**opts)
                    z=flat.candidates(e[ft['req'],ft['local']],flat.pack_nodes(q,ft),
                        flat.pack_nodes(tok,ft),ft,9,n,5,t,t,return_scores=True,**opts)
                    for x,y in zip(a,z):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)

    def test_scalar_probabilities_preserve_fp32_input(self):
        x=torch.randn(2,3,19);saved=x.clone()
        for t in (0.,.1,.7,1.):
            out=rp.probabilities(x,t)
            torch.testing.assert_close(out,(saved/(t or 1)).softmax(-1))
            self.assertTrue(torch.equal(x,saved))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA')
class RootGraphs(unittest.TestCase):
    @torch.inference_mode()
    def test_python_scalar_temperature_is_capture_safe(self):
        x=torch.randn(2,3,19,device='cuda');before=x.clone()
        for t in (.1,.7,1.):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):rp.probabilities(x,t)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):y=rp.probabilities(x,t)
            x.copy_(torch.randn_like(x));graph.replay()
            torch.testing.assert_close(y,(x/t).softmax(-1))
        x.copy_(before)

    @torch.inference_mode()
    def test_native_options_reach_dense_flat_and_chain_graphs(self):
        from ssd.engine.helpers.p2_tree import ChainProxyCUDAGraph
        torch.manual_seed(319);b,n,v=2,4,73
        e=torch.randn(b,n+1,v,device='cuda');q=torch.randn(b,n,v,device='cuda')
        tok=torch.randint(v,(b,n),device='cuda')
        ps=[[-1,0,1,2],[-1,0]];ss=[[0]*4,[0]*2]
        for norm in ('topm','full'):
            cfg,_=make_config(duet_root_normalization=norm)
            opts=rp.options(cfg)
            dg=dense.BatchedTreeProxy(b,n,v,torch.float32,'cuda',8,n,5,policy=opts)
            fg=flat.FlatProxy(b,n,v,torch.float32,'cuda',8,n,5,policy=opts)
            cg=ChainProxyCUDAGraph(n,v,5,8,True,torch.float32,'cuda',policy=opts)
            for temps in ([.1,.7],[.7,1.],[1.,.1]):
                dg.prepare(ps,ss,tok,q,temps,temps);fg.prepare(ps,ss,tok,q,temps,temps)
                topo=fg.current['topo'];a=dg.replay(e);z=fg.replay(e[topo['req'],topo['local']])
                c=cg.replay(e[0],q[0],tok[0],temps[0],temps[0])
                for x,y in zip(a,z):self.assertTrue(torch.equal(x,y))
                for x,y in zip(a,c[:2]):self.assertTrue(torch.equal(x[0],y))


if __name__=='__main__':unittest.main()
