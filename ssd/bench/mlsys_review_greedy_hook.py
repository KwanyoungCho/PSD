"""Benchmark-only full-model adapter for batched greedy host traversal.

Use --preflight-test mlsys_review_greedy_hook with SSD_REVIEW_GREEDY_HOST=0/1.
The adapter adds one CPU-topology assignment to the current verifier source
in this benchmark process, and substitutes only greedy acceptance replay.
All original CUDA captures still initialize, keeping allocation/warmup comparable.
No production source, default or stochastic acceptance is modified.
"""
import inspect
import os
import unittest
import torch
from ssd.engine.helpers import batch_tree_verify as module
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeAccept

ENABLED=os.getenv('SSD_REVIEW_GREEDY_HOST','0')=='1'
ANCHOR='    proxy.prepare(parents,siblings,tokens,q,tt,dt)\n'
SOURCE=inspect.getsource(module.verify)
assert SOURCE.count(ANCHOR)==1, 'Verifier changed; review this adapter before running'

if ENABLED:
    insertion=ANCHOR+'    proxy._review_host_topology=(parents,[row[1:] for row in spec])\n'
    adapted=SOURCE.replace(ANCHOR,insertion)
    exec(compile(adapted,module.__file__+':greedy_host_review','exec'),vars(module))
    original_init=BatchedTreeAccept.__init__
    original_replay=BatchedTreeAccept.replay

    def initialize(self,proxy,sampler_x,fan_out,greedy,pool=None):
        original_init(self,proxy,sampler_x,fan_out,greedy,pool)
        self._review_greedy=greedy

    @torch.inference_mode()
    def replay(self,logits,tt,dt):
        if not self._review_greedy:return original_replay(self,logits,tt,dt)
        parents,tokens=self.proxy._review_host_topology
        argmax=logits.argmax(-1).cpu().tolist();n=self.proxy.tokens.shape[1]
        paths=[];recoveries=[];terminals=[]
        for bi,par in enumerate(parents):
            children={(pa,token):j for j,(pa,token) in enumerate(zip(par,tokens[bi]))}
            path=[];ctx=-1
            for _ in range(len(par)+1):
                token=argmax[bi][ctx+1];child=children.get((ctx,token))
                if child is None:break
                path.append(child);ctx=child
            paths.append(path+[-1]*(n-len(path)));recoveries.append([token]);terminals.append([ctx+1])
        return tuple(torch.tensor(x,dtype=torch.int64) for x in (paths,recoveries,terminals))
    BatchedTreeAccept.__init__=initialize
    BatchedTreeAccept.replay=replay

class AdapterContract(unittest.TestCase):
    def test_opt_in_and_exact_insertion_point(self):
        self.assertEqual(SOURCE.count(ANCHOR),1)
        self.assertEqual(hasattr(BatchedTreeAccept.replay,'__wrapped__'),True)

    def test_mixed_branch_chain_and_empty_match_greedy_reference(self):
        if not ENABLED:self.skipTest('adapter is disabled in this control')
        from types import SimpleNamespace
        from ssd.engine.helpers.batch_tree_sampling import pack_topologies,verify_batch
        parents=[[-1,-1,0,0],[-1,0],[]];siblings=[[0,1,0,1],[0,0],[]]
        tokens=torch.tensor([[1,3,4,6],[2,5,0,0],[0,0,0,0]])
        logits=torch.zeros(3,5,9)
        peaks=torch.tensor([[1,6,7,8,8],[2,5,6,0,0],[8,0,0,0,0]])
        logits.scatter_(2,peaks[:,:,None],5)
        proxy=SimpleNamespace(tokens=tokens,_review_host_topology=(parents,tokens.tolist()))
        receiver=SimpleNamespace(proxy=proxy,_review_greedy=True)
        actual=BatchedTreeAccept.replay(receiver,logits,[0]*3,[0]*3)
        expected=verify_batch(logits,torch.zeros(3,4,9),tokens,
            pack_topologies(parents,siblings,4,'cpu',c_max=3),[0]*3,[0]*3,greedy=True)
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertIn('_review_host_topology',module.verify.__wrapped__.__code__.co_names)
