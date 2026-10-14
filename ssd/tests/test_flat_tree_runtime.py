"""Production compact graph reuse and ordered proposal parity across Q shapes."""
import unittest
import torch
from ssd.engine.helpers import flat_tree_sampling as flat
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CompactGraphReuse(unittest.TestCase):
    @torch.inference_mode()
    def test_vocabulary_shapes_and_rng(self):
        b,n=8,8;dev='cuda';count=0
        for v,dtype in ((32000,torch.float16),(128256,torch.bfloat16)):
          for siblings in (2,3):
            p=BatchedTreeProxy(b,n,v,dtype,dev,25,n,13,c_max=siblings)
            f=flat.FlatProxy(b,n,v,dtype,dev,25,n,13,c_max=siblings)
            accepts={};fa={}
            for lengths in ([8]*4+[2]*4,[8]+[0]*7,[4]*8,[8]*4+[2]*4):
                ps=[[-1,-1,0,0,1,2,3,4][:z] for z in lengths]
                ss=[[0,1,0,1,0,0,0,0][:z] for z in lengths]
                torch.manual_seed(712)
                tokens=torch.arange(1,n+1,device=dev).expand(b,n)
                logits=torch.randn(b,n+1,v,device=dev,dtype=dtype)
                q=torch.randn(b,n,v,device=dev,dtype=dtype)
                for temp in (0.,.3,1.):
                    tt=torch.full((b,),temp,device=dev)
                    p.prepare(ps,ss,tokens,q,tt,tt);f.prepare(ps,ss,tokens,q,tt,tt)
                    top=f.current['topo'];packed=logits[top['req'],top['local']]
                    a=p.replay(logits);z=f.replay(packed)
                    from ssd.engine.helpers.p2_tree import unpack_piv
                    at,ascore=unpack_piv(a[1]);zt,zscore=unpack_piv(z[1]);positive=ascore>1.0001e-6
                    torch.testing.assert_close(ascore,zscore,rtol=1e-5,atol=1e-7)
                    self.assertTrue(torch.equal(a[0][positive],z[0][positive]))
                    self.assertTrue(torch.equal(at[positive],zt[positive]))
                    greedy=temp==0
                    if greedy not in accepts:
                        accepts[greedy]=BatchedTreeAccept(p,None,1,greedy,stochastic_only=True)
                        fa[greedy]=flat.FlatAccept(f,None,1,greedy,stochastic_only=True)
                    else:fa[greedy]._capture()
                    for seed in range(5):
                        torch.cuda.manual_seed(seed)
                        expected=tuple(x.clone() for x in accepts[greedy].replay(logits,tt,tt))
                        torch.cuda.manual_seed(seed)
                        actual=fa[greedy].replay(packed,tt,tt)
                        for x,y in zip(expected,actual):self.assertTrue(torch.equal(x,y))
                        count+=1
            self.assertEqual(f.logit_buffer.numel(),b*(n+1)*v)
        print(f'{count} compact persistent graph/RNG comparisons PASS')

if __name__=='__main__':unittest.main()
