"""P1 coverage must match parent-child topology without changing proposal q."""
import unittest
import torch
from ssd.engine.helpers.p1_tree import build_uniform_p1_roots
from ssd.engine.helpers.p2_tree import q_probs_from_logits


def visibility(parents,device='cpu'):
    p=len(parents)
    rows=torch.eye(p,dtype=torch.uint8,device=device)
    for j in range(1,p):
        rows[j] |= rows[parents[j]]
    return rows


def oracle(logits,tokens,parents,u):
    masked=logits.clone()
    for j in range(1,len(parents)):
        masked[parents[j],tokens[j]]=float('-inf')
    values,ids=masked.topk(u,-1)
    good=values.isfinite()
    return torch.where(good,ids,0),good


class ChildExclusionCPU(unittest.TestCase):
    def run_roots(self,logits,tokens,parents,u,temps=None,**kwargs):
        rows=visibility(parents,logits.device)
        return build_uniform_p1_roots(logits,tokens,u,
            torch.ones(len(parents),device=logits.device) if temps is None else temps,
            sampler_x=None,async_fan_out=3,context_glue_rows=rows,**kwargs)

    def test_siblings_unrelated_tokens_and_leaf(self):
        tokens=torch.tensor([0,1,2,3]);parents=[-1,0,0,1]
        logits=torch.tensor([[0.,10.,9.,8.,7.,6.,5.,4.],
                             [0.,6.,10.,9.,8.,7.,5.,4.],
                             [0.,6.,7.,10.,9.,8.,5.,4.],
                             [0.,1.,2.,3.,4.,5.,6.,7.]])
        before=logits.clone()
        result=self.run_roots(logits,tokens,parents,2)
        self.assertEqual(result['tokens'].view(4,2).tolist(),[[3,4],[2,4],[3,4],[7,6]])
        self.assertTrue(torch.equal(logits,before))

    def test_same_token_at_another_prefix_remains_eligible(self):
        # Token 7 is an existing child of A, not of its sibling B.
        parents=[-1,0,0,1];tokens=torch.tensor([0,1,2,7])
        logits=torch.arange(32,dtype=torch.float32).reshape(4,8)
        result=self.run_roots(logits,tokens,parents,1)
        self.assertEqual(result['tokens'].tolist(),[7,6,7,7])

    def test_original_q_and_context_reach_are_not_renormalized(self):
        g=torch.Generator().manual_seed(14)
        logits=torch.randn(6,43,generator=g)
        parents=[-1,0,0,1,1,2];tokens=torch.tensor([0,3,4,5,6,7])
        temps=torch.tensor([.4,.7,1.,1.2,.8,.9])
        result=self.run_roots(logits,tokens,parents,3,temps)
        full=q_probs_from_logits(logits,temps,None,3)
        reach=torch.ones(6)
        for j in range(1,6):reach[j]=reach[parents[j]]*full[parents[j],tokens[j]]
        ids=result['tokens'].view(6,3)
        expected=(full.gather(1,ids)*reach[:,None]).reshape(-1)
        self.assertTrue(torch.allclose(result['scores'],expected,atol=2e-7,rtol=2e-6))
        self.assertTrue(torch.allclose(result['context_reach'],reach,atol=2e-7,rtol=2e-6))

    def test_random_topologies_match_full_vocabulary_oracle(self):
        g=torch.Generator().manual_seed(23)
        for p in (1,2,4,6,16,24):
            for repeat in range(8):
                parents=[-1]+[int(torch.randint(j,(),generator=g)) for j in range(1,p)]
                logits=torch.randn(p,53,generator=g)
                tokens=torch.randint(53,(p,),generator=g)
                result=self.run_roots(logits,tokens,parents,4)
                ids,good=oracle(logits,tokens,parents,4)
                self.assertTrue(torch.equal(result['tokens'].view(p,4),ids),(p,repeat))
                self.assertTrue(torch.equal(result['valid'].view(p,4),good))

    def test_chain_implicit_and_explicit_visibility_agree(self):
        g=torch.Generator().manual_seed(9)
        logits=torch.randn(11,53,generator=g);tokens=torch.randint(53,(11,),generator=g)
        a=build_uniform_p1_roots(logits,tokens,4,torch.ones(11),sampler_x=None,async_fan_out=3)
        b=self.run_roots(logits,tokens,[-1]+list(range(10)),4)
        for name in ('tokens','scores','valid','context_reach','glue_rows'):
            self.assertTrue(torch.equal(a[name],b[name]),name)

    def test_exhausted_tiny_vocabulary_is_invalid_padding(self):
        parents=[-1,0,0,0];tokens=torch.tensor([0,0,1,2])
        logits=torch.tensor([[4.,3.,2.,1.]]*4)
        out=self.run_roots(logits,tokens,parents,2,root_width=10)
        self.assertEqual(out['tokens'][:2].tolist(),[3,0])
        self.assertEqual(out['valid'].tolist(),[True,False]+[True]*6+[False]*2)
        self.assertEqual(float(out['scores'][1]),0.)

    def test_equal_scores_choose_eligible_tokens_with_maximum_values(self):
        parents=[-1,0,0,1];tokens=torch.tensor([0,1,2,3])
        logits=torch.ones(4,8)
        out=self.run_roots(logits,tokens,parents,3)
        ids=out['tokens'].view(4,3).tolist()
        for parent,row in enumerate(ids):
            excluded={int(tokens[j]) for j in range(1,4) if parents[j]==parent}
            self.assertFalse(set(row)&excluded)
            self.assertEqual(len(set(row)),3)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ChildExclusionCUDA(unittest.TestCase):
    def test_native_context_sizes_reuse_compiled_selector(self):
        from ssd.engine.helpers.p1_root_selection import select_root_tokens, _filter_roots
        def run(p):
            logits=torch.randn(p,32000,device='cuda',dtype=torch.bfloat16)
            tokens=torch.arange(p,device='cuda')
            parents=torch.zeros(p,device='cuda',dtype=torch.int64)
            select_root_tokens(logits,tokens,parents,4)
        run(6)
        cache=_filter_roots.device_caches[torch.cuda.current_device()][0]
        before=len(cache)
        for p in range(1,25):run(p)
        self.assertEqual(len(cache),before,'Live context length triggered another JIT specialization')

    def test_maximum_fanout_and_exhaustion_mask(self):
        for p,v,u in ((24,32000,4),(4,4,2)):
            parents=[-1]+[0]*(p-1)
            tokens=torch.tensor([0]+list(range(v-1,v-p,-1)),device='cuda')
            logits=torch.arange(v,dtype=torch.float32,device='cuda').expand(p,v).contiguous()
            out=build_uniform_p1_roots(logits,tokens,u,torch.tensor([.7],device='cuda'),
                sampler_x=None,async_fan_out=5,context_glue_rows=visibility(parents,'cuda'))
            ids,valid=oracle(logits,tokens,parents,u)
            self.assertTrue(torch.equal(out['tokens'].view(p,u),ids))
            self.assertTrue(torch.equal(out['valid'].view(p,u),valid))

    def test_cuda_matches_cpu_for_all_native_shapes(self):
        g=torch.Generator().manual_seed(41)
        for p in (1,5,6,11,16,24):
            parents=[-1]+[int(torch.randint(j,(),generator=g)) for j in range(1,p)]
            tokens=torch.randint(32000,(p,),generator=g)
            logits=torch.randn(p,32000,generator=g).bfloat16()
            def run(dev):
                return build_uniform_p1_roots(logits.to(dev),tokens.to(dev),4,
                    torch.tensor([.7],device=dev),sampler_x=None,async_fan_out=5,
                    context_glue_rows=visibility(parents,dev))
            cpu=run('cpu');gpu=run('cuda')
            # CPU and CUDA topk have different permissible tie order.
            for i,row in enumerate(gpu['tokens'].view(p,4).cpu().tolist()):
                excluded={int(tokens[j]) for j in range(1,p) if parents[j]==i}
                self.assertFalse(set(row)&excluded)
                masked=logits[i].float().clone()
                for t in excluded:masked[t]=float('-inf')
                self.assertTrue(torch.equal(masked[row].sort(descending=True).values,
                                            masked.topk(4).values))
            # The selected-q kernel has the same FP32 proposal semantics.
            ids=gpu['tokens'].view(p,4).cpu()
            q=q_probs_from_logits(logits.float(),torch.full((p,),.7),None,5)
            reach=cpu['context_reach']
            expected=(q.gather(1,ids)*reach[:,None]).reshape(-1)
            self.assertTrue(torch.allclose(gpu['scores'].cpu(),expected,atol=2e-7,rtol=3e-5))

    def test_graph_replay_uses_live_parent_and_token_inputs(self):
        p=6;u=4;dev='cuda'
        logits=torch.arange(p*53,dtype=torch.float32,device=dev).view(p,53)
        tokens=torch.tensor([0,48,49,50,51,52],device=dev)
        rows=visibility([-1,0,0,1,1,2],dev)
        temps=torch.tensor([.7],device=dev)
        buffers=dict(tokens=torch.empty(26,dtype=torch.int64,device=dev),
            scores=torch.empty(26,device=dev),context_ids=torch.empty(26,dtype=torch.int64,device=dev),
            valid=torch.empty(26,dtype=torch.bool,device=dev),glue_rows=torch.empty(26,p,dtype=torch.uint8,device=dev),
            context_reach=torch.empty(p,device=dev))
        def run():return build_uniform_p1_roots(logits,tokens,u,temps,sampler_x=None,
            async_fan_out=5,context_glue_rows=rows,root_width=26,output_buffers=buffers)
        for _ in range(3):run()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=run()
        for parents in ([-1,0,0,0,2,3],[-1,0,1,2,3,4]):
            rows.copy_(visibility(parents,dev));tokens.add_(1).remainder_(53)
            expected,_=oracle(logits,tokens,parents,u)
            graph.replay();torch.cuda.synchronize()
            self.assertTrue(torch.equal(out['tokens'][:p*u].view(p,u),expected))
            for key in buffers:self.assertEqual(out[key].data_ptr(),buffers[key].data_ptr())
            self.assertFalse(bool(out['valid'][p*u:].any()))


if __name__=='__main__':unittest.main()
