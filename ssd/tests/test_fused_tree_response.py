"""Response packet, ancestry, first-match selection, and RNG transition checks."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from ssd.engine.helpers.response_topology import prepare_response_topology
from ssd.engine.helpers.response_snapshot import parse_host_response
from ssd.engine.helpers.fused_tree_response import attach_response_packet, try_response, glue_context


def views():
    return dict(tok=torch.tensor([[4,5,6],[7,8,9]]),
        parent_local=torch.tensor([[-1,-1,0],[-1,0,1]]),
        sib_order=torch.tensor([[0,1,0],[0,0,0]]),
        parent_q_ref=torch.tensor([[0,0,1],[0,1,2]]),
        valid=torch.tensor([3,3]),u_valid=torch.tensor([2,3]),
        parent_q_cells=torch.tensor([[1,2,0],[0,1,2]]),
        cell_logits=torch.arange(48,dtype=torch.float32).view(3,16))


class Response(unittest.TestCase):
    def fixture(self,phase=1):
        v=views();snap=prepare_response_topology(v,3)
        v['_response_topology']=snap
        v['_response_packet']=attach_response_packet(snap,phase,5)
        c=SimpleNamespace(use_eagle=False,duet_enabled=True,verbose=False,
            duet_proxy_on_draft=False,duet_response_token_width=5,
            duet_p1_tree_verify_nodes=3,duet_p2_tree_verify_nodes=3)
        keys=torch.tensor([[10,2,13],[10,2,13]])
        r=SimpleNamespace(config=c,tree_cache_keys=keys,_last_n_draft_keys=2 if phase==1 else 0,
            _p1_tree_views=v if phase==1 else None,_tree_views=v if phase==2 else None,
            hf_config=SimpleNamespace(vocab_size=16,torch_dtype=torch.float32),device='cpu',
            _tree_wire_parent_q_buf=torch.empty(1,3,16))
        return r,v,keys[:1].clone()

    def test_packet_layout_padding_and_cpu_snapshot(self):
        v=views();s=prepare_response_topology(v,3)
        for p in (1,2):
            buf=attach_response_packet(s,p,5)
            for root in range(2):
                host=parse_host_response(buf[root].tolist(),1,5,15)
                self.assertEqual(host.cache_hits,(1,));self.assertEqual(host.phase_source,(p,))
                self.assertEqual(host.valid_k,(3,));self.assertEqual(host.tokens[0][:3],tuple(v['tok'][root].tolist()))
                self.assertEqual(host.tokens[0][3:],(0,0))
                self.assertEqual(host.tree_ints[0],s.integer_rows[root])

    def test_single_readback_keeps_first_matching_root_and_rng(self):
        def gather(source,index,out):out.copy_(source[index])
        for phase in (1,2):
            r,v,key=self.fixture(phase)
            torch.manual_seed(87)
            expected_q=torch.empty((1,4,16)).uniform_();state=torch.get_rng_state().clone()
            torch.manual_seed(87)
            with patch('ssd.engine.helpers.p2_tree_executor._gather_backbone_logits',side_effect=gather):
                result=try_response(r,key,1,4,torch.tensor([33]))
            self.assertIsNotNone(result)
            self.assertTrue(torch.equal(result[1],expected_q))
            self.assertTrue(torch.equal(torch.get_rng_state(),state))
            self.assertEqual(r._tree_hit_root,0);self.assertEqual(r._tree_hit_phase,phase)
            self.assertEqual(r._tree_served_numtok,33);self.assertEqual(r._tree_served_seq,10)
            self.assertTrue(torch.equal(result[0][0,:3],v['tok'][0]))
            self.assertTrue(torch.equal(r._tree_wire_parent_q_buf[0],v['cell_logits'][v['parent_q_cells'][0]]))
            self.assertIsNone(result[2])

    def test_miss_and_non_tree_fallback_do_not_consume_rng(self):
        r,v,k=self.fixture()
        for keys in (k+99,k):
            if torch.equal(keys,k):r._p1_tree_views=None
            before=torch.get_rng_state().clone()
            self.assertIsNone(try_response(r,keys,1,4,torch.tensor([33])))
            self.assertTrue(torch.equal(torch.get_rng_state(),before))

    def test_chain_only_skips_extra_lookup(self):
        r,v,k=self.fixture()
        r._p1_tree_views=None
        r.tree_cache_keys=SimpleNamespace(numel=lambda:2)
        # The sentinel cannot participate in a GPU comparison. A chain-only
        # cache must return before attempting that extra classification.
        self.assertIsNone(try_response(r,k,1,4,torch.tensor([33])))

    def test_snapshot_remains_immutable_after_executor_reuse(self):
        r,v,k=self.fixture();old=v['_response_packet'].clone()
        v['tok'].fill_(15);v['valid'].fill_(0)
        self.assertTrue(torch.equal(v['_response_packet'],old))

    def test_ancestry_matches_reference_for_branching_and_deep_trees(self):
        for par in ([-1],[-1,-1,0,0,1],[-1,0,1,2,3],[-1,-1,-1]):
            depth,anc,rows,ctx=glue_context(par)
            expected=np.zeros((len(par)+1,len(par)+1),dtype=np.uint8);expected[:,0]=1
            for j in range(len(par)):
                cur=j;path=[]
                while cur>=0:path.append(cur);cur=par[cur]
                expected[j+1,1+np.asarray(path)]=1
                self.assertEqual(ctx[j+1],len(path));self.assertEqual(depth[j]+1,len(path))
                self.assertEqual(anc[j],list(reversed(path[1:])))
            np.testing.assert_array_equal(rows,expected)


if __name__=='__main__':unittest.main()
