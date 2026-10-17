"""Check input overwrite equivalence and immutable validation-cache gates."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from ssd.engine.model_runner import ModelRunner
from ssd.engine.helpers.p2_tree import parse_tree_ints,validate_tree_ints,TreeProxyCUDAGraph


class ValidationCache(unittest.TestCase):
    def runner(self,wire,cached):
        return SimpleNamespace(config=SimpleNamespace(duet_tree_wire_nodes=3,
            hf_config=SimpleNamespace(vocab_size=32),duet_tree_c_tensor=3),
            _duet_tree_meta=wire,_duet_validated_tree=cached,
            _duet_verify_context_len_cpu=0)

    def wire(self):return tuple([1,1,1]+[5,0,0]+[-1,-1,-1]+[0,0,0]+[0,0,0])

    def test_same_immutable_packet_reuses_prior_validation(self):
        w=self.wire();r=self.runner(w,(w,parse_tree_ints(w,3)))
        with patch('ssd.engine.helpers.p2_tree.parse_tree_ints',wraps=parse_tree_ints) as p,patch('ssd.engine.helpers.p2_tree.validate_tree_ints',wraps=validate_tree_ints) as v:
            with self.assertRaisesRegex(RuntimeError,'invalid KV geometry'):
                ModelRunner._run_tree_verify(r,torch.zeros(2,dtype=torch.long),None,False)
            p.assert_not_called();v.assert_not_called()

    def test_other_packet_and_mutable_packet_validate_normally(self):
        for mutable in (False,True):
            old=self.wire();w=list(old) if mutable else tuple(list(old))
            r=self.runner(w,(old,parse_tree_ints(old,3)))
            with patch('ssd.engine.helpers.p2_tree.parse_tree_ints',wraps=parse_tree_ints) as p,patch('ssd.engine.helpers.p2_tree.validate_tree_ints',wraps=validate_tree_ints) as v:
                with self.assertRaisesRegex(RuntimeError,'invalid KV geometry'):
                    ModelRunner._run_tree_verify(r,torch.zeros(2,dtype=torch.long),None,False)
                p.assert_called_once();v.assert_called_once()

    def test_malformed_different_packet_is_rejected(self):
        old=self.wire();w=list(old);w[6]=12
        r=self.runner(tuple(w),(old,parse_tree_ints(old,3)))
        with self.assertRaises((RuntimeError,ValueError)):
            ModelRunner._run_tree_verify(r,torch.zeros(2,dtype=torch.long),None,False)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ExactOverwrite(unittest.TestCase):
    @torch.inference_mode()
    def test_full_and_partial_replay_match_old_zero_then_copy(self):
        kwargs=dict(nv=4,vocab_size=32,wire_n=8,depth_steps=4,dtype=torch.float32,device='cuda:0',top_k=4,c_max=3)
        a=TreeProxyCUDAGraph(**kwargs);b=TreeProxyCUDAGraph(**kwargs)
        torch.manual_seed(93)
        for valid in (4,4,2,4,1):
            par=[-1,0,1,2][:valid];sib=[0]*valid
            a.prepare_topology(par,sib);b.prepare_topology(par,sib)
            x=torch.randn(valid+1,32,device='cuda:0');q=torch.randn(valid,32,device='cuda:0')
            tokens=torch.arange(1,valid+1,device='cuda:0')
            # Reference uses zero-before-copy; candidate starts with garbage.
            a.in_exit.zero_();a.in_q.zero_();a.in_tokens.zero_()
            b.in_exit.fill_(99);b.in_q.fill_(88);b.in_tokens.fill_(17)
            expected=a.replay(x,q,tokens,.7,.7);actual=b.replay(x,q,tokens,.7,.7)
            for left,right in zip(expected,actual):self.assertTrue(torch.equal(left,right))
            self.assertTrue(torch.equal(a.in_exit,b.in_exit));self.assertTrue(torch.equal(a.in_q,b.in_q))
            self.assertTrue(torch.equal(a.in_tokens,b.in_tokens))


if __name__=='__main__':unittest.main(verbosity=2)
