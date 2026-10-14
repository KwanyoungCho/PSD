"""Cross-request forest isolation and exact law checks for review kernels."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import torch
import mlsys_flat_tree as flat
from ssd.engine.helpers.batch_tree_sampling import pack_topologies, ladder, candidates, verify_batch


def test_flat_reference(lengths,temperature):
    torch.manual_seed(72)
    b=len(lengths);n=8;v=31
    parent=[-1,-1,0,0,1,2,3,4];sibling=[0,1,0,1,0,0,0,0]
    ps=[parent[:z] for z in lengths];ss=[sibling[:z] for z in lengths]
    dense=pack_topologies(ps,ss,n,'cpu');f=flat.topology(ps,ss,n,3,'cpu',sum(lengths)+b+3)
    tokens=torch.arange(1,n+1).expand(b,n).clone()
    lp=torch.randn(b,n+1,v);lq=torch.randn(b,n,v)
    qlp=lp[f['req'],f['local']];qlq=flat.pack_nodes(lq,f);qt=flat.pack_nodes(tokens,f)
    t=torch.full((b,),temperature)
    pp=(lp/max(temperature,1e-10)).softmax(-1);qq=(lq/max(temperature,1e-10)).softmax(-1)
    a,term,r=ladder(tokens,pp,qq,dense,n)
    fa,ft,fr=flat.ladder(qt,pp[f['req'],f['local']],flat.pack_nodes(qq,f),f,n)
    valid=f['valid']
    torch.testing.assert_close(fa[valid],a[f['req'][valid],f['local'][valid]])
    torch.testing.assert_close(ft[valid],term[f['req'][valid],f['local'][valid]])
    torch.testing.assert_close(fr[valid],r[f['req'][valid],f['local'][valid]])
    # Compare positive candidates; zero-score padding has no ranking contract.
    x=candidates(lp,lq,tokens,dense,7,n,5,t,t,
        source="residual",normalization="topm",overlap_mix=0.)
    y=flat.candidates(qlp,qlq,qt,f,7,n,5,t,t)
    from ssd.engine.helpers.p2_tree import unpack_piv
    tx,sx=unpack_piv(x[1]);ty,sy=unpack_piv(y[1]);positive=sx>1.0001e-6
    torch.testing.assert_close(sx,sy)
    assert torch.equal(x[0][positive],y[0][positive])
    assert torch.equal(tx[positive],ty[positive])
    coins=torch.rand(b,n)
    torch.manual_seed(83)
    x=verify_batch(lp,lq,tokens,dense,t,t,greedy=temperature==0,uniforms=coins,stochastic_only=True)
    torch.manual_seed(83)
    y=flat.verify(qlp,qlq,qt,f,n,t,t,greedy=temperature==0,uniforms=coins)
    for a,z in zip(x,y):torch.testing.assert_close(a,z)
    # Changing request 0's distribution must not alter other requests' laws.
    changed=qlp.clone();changed[f['req']==0]=torch.randn_like(changed[f['req']==0])*4
    _,_,before=flat.ladder(qt,qlp.softmax(-1),qlq.softmax(-1),f,n,exact=True)
    _,_,after=flat.ladder(qt,changed.softmax(-1),qlq.softmax(-1),f,n,exact=True)
    torch.testing.assert_close(before[f['req']!=0],after[f['req']!=0])

class FlatForestLaw(unittest.TestCase):
    def test_reference_and_request_isolation(self):
        for lengths in [(8,2,0),(0,0,0),(8,8),(1,5,3)]:
            for temp in [0.,.3,1.]:
                with self.subTest(lengths=lengths, temperature=temp):
                    test_flat_reference(lengths,temp)

if __name__=="__main__":
    unittest.main()
