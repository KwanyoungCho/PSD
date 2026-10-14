"""Removing absent sibling stages must preserve root wire and exact sampling."""
import unittest
import torch
from ssd.engine.helpers.batch_tree_sampling import pack_topologies,candidates,verify_batch

class LiveSiblingLaw(unittest.TestCase):
    def test_exact_law_and_wire(self):
        for c,ps,ss in [(1,[-1,0,1,2],[0,0,0,0]),(2,[-1,-1,0,1],[0,1,0,0])]:
            for t in [0.,.7,1.]:
                for seed in range(20):
                    torch.manual_seed(seed)
                    lp=torch.randn(3,5,31);lq=torch.randn(3,4,31)
                    tok=torch.arange(4).expand(3,4);temp=torch.full((3,),t)
                    wide=pack_topologies([ps]*3,[ss]*3,4,'cpu',3)
                    slim=pack_topologies([ps]*3,[ss]*3,4,'cpu',c)
                    x=candidates(lp,lq,tok,wide,7,4,5,temp,temp)
                    y=candidates(lp,lq,tok,slim,7,4,5,temp,temp)
                    for a,b in zip(x,y):self.assertTrue(torch.equal(a,b))
                    torch.manual_seed(82)
                    x=verify_batch(lp,lq,tok,wide,temp,temp,greedy=t==0,stochastic_only=True)
                    torch.manual_seed(82)
                    y=verify_batch(lp,lq,tok,slim,temp,temp,greedy=t==0,stochastic_only=True)
                    for a,b in zip(x,y):self.assertTrue(torch.equal(a,b))

if __name__=='__main__':unittest.main()
